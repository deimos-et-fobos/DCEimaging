//   complex.h - declaration of class
//   of complex number
//
//   The code is property of LIBROW
//   You can use it on your own
//   When utilizing credit LIBROW site

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef _COMPLEX_H_
#define _COMPLEX_H_

class complex
{
protected:
	//   Internal presentation - real and imaginary parts
	float m_re;
	float m_im;

public:
	//   Imaginary unity
	static const complex i;
	static const complex j;

	//   Constructors
	CUDA_CALLABLE_MEMBER complex(): m_re(0.), m_im(0.) {}
	CUDA_CALLABLE_MEMBER complex(float re, float im): m_re(re), m_im(im) {}
	CUDA_CALLABLE_MEMBER complex(float val): m_re(val), m_im(0.) {}

	//   Assignment
	CUDA_CALLABLE_MEMBER complex& operator= (const complex c1){
		this->m_re = c1.re();
		this->m_im = c1.im();
		return *this;
	}
	CUDA_CALLABLE_MEMBER complex& operator= (const float val){
		m_re = val;
		m_im = 0.;
		return *this;
	}
	CUDA_CALLABLE_MEMBER void setvalues(const float val){	m_re = val; }
  CUDA_CALLABLE_MEMBER void setvalues(const float re,const float im){
    m_re = re;
    m_im = im;
  }

	//   Basic operations - taking parts
	CUDA_CALLABLE_MEMBER float re() const{ return m_re; }
	CUDA_CALLABLE_MEMBER float im() const{ return m_im; }

	//   Conjugate number
	CUDA_CALLABLE_MEMBER complex conjugate() const{	return complex(m_re, -m_im); }

	//   Norm   
	CUDA_CALLABLE_MEMBER float norm() const{	return m_re * m_re + m_im * m_im;	}

	//   Arithmetic operations
	CUDA_CALLABLE_MEMBER complex operator+ (const complex& other) const{
		return complex(m_re + other.m_re, m_im + other.m_im);
	}

	CUDA_CALLABLE_MEMBER complex operator- (const complex& other) const{
		return complex(m_re - other.m_re, m_im - other.m_im);
	}

	CUDA_CALLABLE_MEMBER complex operator* (const complex& other) const{
		return complex(m_re * other.m_re - m_im * other.m_im,
			m_re * other.m_im + m_im * other.m_re);
	}

	CUDA_CALLABLE_MEMBER complex operator/ (const complex& other) const{
		const float denominator = other.m_re * other.m_re + other.m_im * other.m_im;
		return complex((m_re * other.m_re + m_im * other.m_im) / denominator,
			(m_im * other.m_re - m_re * other.m_im) / denominator);
	}

	CUDA_CALLABLE_MEMBER complex& operator+= (const complex& other){
		m_re += other.m_re;
		m_im += other.m_im;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator-= (const complex& other){
		m_re -= other.m_re;
		m_im -= other.m_im;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator*= (const complex& other){
		const float temp = m_re;
		m_re = m_re * other.m_re - m_im * other.m_im;
		m_im = m_im * other.m_re + temp * other.m_im;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator/= (const complex& other){
		const float denominator = other.m_re * other.m_re + other.m_im * other.m_im;
		const float temp = m_re;
		m_re = (m_re * other.m_re + m_im * other.m_im) / denominator;
		m_im = (m_im * other.m_re - temp * other.m_im) / denominator;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator++ (){
		++m_re;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex operator++ (int){
		complex temp(*this);
		++m_re;
		return temp;
	}

	CUDA_CALLABLE_MEMBER complex& operator-- (){
		--m_re;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex operator-- (int){
		complex temp(*this);
		--m_re;
		return temp;
	}

	CUDA_CALLABLE_MEMBER complex operator+ (const float val) const{
		return complex(m_re + val, m_im);
	}

	CUDA_CALLABLE_MEMBER complex operator- (const float val) const{
		return complex(m_re - val, m_im);
	}

	CUDA_CALLABLE_MEMBER complex operator* (const float val) const{
		return complex(m_re * val, m_im * val);
	}

	CUDA_CALLABLE_MEMBER complex operator/ (const float val) const{
		return complex(m_re / val, m_im / val);
	}

	CUDA_CALLABLE_MEMBER complex& operator+= (const float val){
		m_re += val;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator-= (const float val){
		m_re -= val;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator*= (const float val){
		m_re *= val;
		m_im *= val;
		return *this;
	}

	CUDA_CALLABLE_MEMBER complex& operator/= (const float val){
		m_re /= val;
		m_im /= val;
		return *this;
	}

	CUDA_CALLABLE_MEMBER friend complex operator+ (const float left, const complex& right){
		return complex(left + right.m_re, right.m_im);
	}

	CUDA_CALLABLE_MEMBER friend complex operator- (const float left, const complex& right){
		return complex(left - right.m_re, -right.m_im);
	}

	CUDA_CALLABLE_MEMBER friend complex operator* (const float left, const complex& right){
		return complex(left * right.m_re, left * right.m_im);
	}

	CUDA_CALLABLE_MEMBER friend complex operator/ (const float left, const complex& right){
		const float denominator = right.m_re * right.m_re + right.m_im * right.m_im;
		return complex(left * right.m_re / denominator,
			-left * right.m_im / denominator);
	}

	//   Boolean operators
	CUDA_CALLABLE_MEMBER bool operator== (const complex &other) const{
		return m_re == other.m_re && m_im == other.m_im;
	}

	CUDA_CALLABLE_MEMBER bool operator!= (const complex &other) const{
		return m_re != other.m_re || m_im != other.m_im;
	}

	CUDA_CALLABLE_MEMBER bool operator== (const float val) const{
		return m_re == val && m_im == 0.;
	}

	CUDA_CALLABLE_MEMBER bool operator!= (const float val) const{
		return m_re != val || m_im != 0.;
	}

	CUDA_CALLABLE_MEMBER friend bool operator== (const float left, const complex& right){
		return left == right.m_re && right.m_im == 0.;
	}

	CUDA_CALLABLE_MEMBER friend bool operator!= (const float left, const complex& right)
	{
		return left != right.m_re || right.m_im != 0.;
	}
};

#endif
