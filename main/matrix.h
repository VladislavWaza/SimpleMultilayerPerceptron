#pragma once
class Matrix
{
	size_t m_row = 0;
	size_t m_col = 0;
	double** m_ptr = nullptr;
public:
	Matrix(size_t row, size_t col);
	Matrix(size_t row, size_t col, double value);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;
	~Matrix();

	double& at(size_t i, size_t j);
	double& operator()(size_t i, size_t j) noexcept;
	double at(size_t i, size_t j) const;
	double operator()(size_t i, size_t j) const noexcept;
	size_t rows() const;
	size_t cols() const;
		
	void rand();
	Matrix trans() const;
};

class Vector
{
	size_t m_size = 0;
	double* m_ptr = nullptr;
public:
	Vector(size_t size);
	Vector(size_t size, double value);
	Vector(const Vector& other);
	Vector(Vector&& other) noexcept;
	~Vector();
	double& at(size_t i);
	double& operator()(size_t i) noexcept;
	double at(size_t i) const;
	double operator()(size_t i) const noexcept;
	size_t size() const;
	void rand();
	Vector& operator+= (const Vector& b);
	Vector& operator= (const Vector& b);
	Vector operator+(const Vector& b) const;
};

Vector operator* (const Matrix& a, const Vector& b);