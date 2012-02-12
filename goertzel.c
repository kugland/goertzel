/*
 *	Copyright (c) 2012 André von Kugland
 *
 *	Permission is  hereby  granted,  free of charge,  to any person
 *	obtaining a copy of this software  and associated documentation
 *	files  (the  "Software"),  to  deal  in  the  Software  without
 *	restriction,  including  without limitation the  rights to use,
 *	copy, modify,  merge, publish, distribute,  sublicense,  and/or
 *	sell copies of the Software,  and to permit persons to whom the
 *	Software  is  furnished  to do  so,  subject  to the  following
 *	conditions:
 *
 *	The above copyright notice  and this permission notice shall be
 *	included in all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *	OF  MERCHANTABILITY,  FITNESS  FOR  A  PARTICULAR  PURPOSE  AND
 *	NONINFRINGEMENT.  IN NO  EVENT SHALL THE  AUTHORS OR  COPYRIGHT
 *	HOLDERS  BE LIABLE FOR ANY  CLAIM,  DAMAGES OR OTHER LIABILITY,
 *	WHETHER IN AN  ACTION OF CONTRACT, TORT  OR OTHERWISE,  ARISING
 *	FROM,  OUT OF OR IN CONNECTION  WITH THE SOFTWARE OR THE USE OR
 *	OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 *	The Goertzel algorithm is a digital signal processing (DSP) technique for
 *	identifying frequency components of a signal, published by Gerald Goertzel
 *	in 1958. While the general Fast Fourier transform algorithm computes evenly
 *	across the bandwidth of the incoming signal, the Goertzel algorithm looks
 *	at specific, predetermined frequencies.
 *
 *	A practical application of this algorithm is recognition of the DTMF tones
 *	produced by the buttons pushed on a telephone keypad.
 *
 *	cf. https://en.wikipedia.org/wiki/Goertzel_algorithm
 */

#if __SSE__
#define GOERTZEL_SSE 1
#endif

#include <math.h>

#if GOERTZEL_SSE
#include <xmmintrin.h>

static void goertzel_sse(size_t, const float *, float,
                         size_t, const float *, float *);
#endif

static float compute_coeff(float n, float f, float sr);

/**
  * @param  nf        number of frequencies to look for
  * @param  f[in]     list of frequencies to look for
  * @param  sr        sample rate
  * @param  n         size of window
  * @param  w[in]     window
  * @param  res[out]  result
  */
void goertzel(size_t nf, const float *f, float sr,
              size_t n,  const float *w, float *r)
{
	int i;

#if GOERTZEL_SSE
	if (nf >= 4) {
		size_t sse_nf = nf & ~3;

		goertzel_sse(sse_nf, f, sr, n, w, r);
		r += sse_nf;
		f += sse_nf;

		nf &= 3;
	}
#endif

	for (i = 0; i < nf; i++) {
		register int j;
		register float coeff, q1, q2;

		q1 = q2 = 0.0f;
		coeff = compute_coeff(n, f[i], sr);

		for (j = 0; j < n; j++) {
			register float q0;

			q0 = coeff * q1 - q2 + w[j];
			q2 = q1;
			q1 = q0;
		}
		r[i] = q1*q1 + q2*q2 - q1*q2*coeff;
	}
}

/**
 * @brief Computes coefficients for goertzel().
 *
 * @param  n   window size
 * @param  f   frequency
 * @param  sr  sample rate
 */
static float compute_coeff(float n, float f, float sr)
{
	return 2.0 * cos((2.0*M_PI/n) * (0.5+(n*f/sr)));
}

#if GOERTZEL_SSE
static void goertzel_sse(size_t nf, const float *f, float sr,
                         size_t n,  const float *w, float *r)
{
	int i;
	float coeff[4];

	for (i = 0; i < nf; i += 4) {
		register int j;
		__m128 q1, q2, coeffv, rv;

		for (j = 0; j < 4; j++)
			coeff[j] = compute_coeff(n, f[i+j], sr);

		coeffv = _mm_loadu_ps(coeff);
		q1 = _mm_xor_ps(q1, q1);
		q2 = _mm_xor_ps(q2, q2);

		for (j = 0; j < n; j++) {
			register __m128 wv, q0;

			wv = _mm_load1_ps(&w[j]);
			q0 = _mm_mul_ps(coeffv, q1);
			q0 = _mm_sub_ps(q0, q2);
			q0 = _mm_add_ps(q0, wv);
			q2 = q1;
			q1 = q0;
		}

		rv = _mm_mul_ps(q1, q2);
		rv = _mm_mul_ps(rv, coeffv);
		q1 = _mm_mul_ps(q1, q1);
		rv = _mm_sub_ps(q1, rv);
		q2 = _mm_mul_ps(q2, q2);
		rv = _mm_add_ps(rv, q2);

		_mm_storeu_ps(&r[i], rv);
	}
}
#endif
