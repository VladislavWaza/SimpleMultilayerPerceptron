#include "matrix.h"
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

Matrix::Matrix(size_t row, size_t col)
{
	m_row = row;
	m_col = col;
	m_ptr = new double* [m_row];
	for (size_t i = 0; i < m_row; ++i)
		m_ptr[i] = new double [m_col];
}

Matrix::Matrix(size_t row, size_t col, double value) : Matrix(row, col)
{
	for (size_t i = 0; i < m_row; ++i)
		for (size_t j = 0; j < m_col; ++j)
			m_ptr[i][j] = value;
}

Matrix::Matrix(const Matrix& other) : Matrix(other.m_row, other.m_col)
{
	for (size_t i = 0; i < m_row; ++i)
		std::memcpy(m_ptr[i], other.m_ptr[i], m_col * sizeof(double));
}

Matrix::Matrix(Matrix&& other) noexcept : m_row(other.m_row), m_col(other.m_col), m_ptr(other.m_ptr)
{
	other.m_ptr = nullptr;
	other.m_row = 0;
	other.m_col = 0;
}

Matrix::~Matrix()
{
	for (size_t i = 0; i < m_row; ++i)
		delete[] m_ptr[i];
	delete[] m_ptr;
}

double& Matrix::at(size_t i, size_t j)
{
	if (i >= m_row || j >= m_col)
		throw std::out_of_range("Out of range in Matrix");
	return m_ptr[i][j];
}

double& Matrix::operator()(size_t i, size_t j) noexcept
{
	return m_ptr[i][j];
}

double Matrix::at(size_t i, size_t j) const
{
	if (i >= m_row || j >= m_col)
		throw std::out_of_range("Out of range in Matrix\n");
	return m_ptr[i][j];
}

double Matrix::operator()(size_t i, size_t j) const noexcept
{
	return m_ptr[i][j];
}

size_t Matrix::rows() const
{
	return m_row;
}

size_t Matrix::cols() const
{
	return m_col;
}

void Matrix::rand()
{
	std::srand(std::time(nullptr));
	for (size_t i = 0; i < m_row; ++i)
		for (size_t j = 0; j < m_col; ++j)
			m_ptr[i][j] = (std::rand() % 1000) * 0.017 / (m_row + m_col + 42);
}

Matrix Matrix::trans() const
{
	Matrix res(this->m_col, this->m_row);

	for (size_t i = 0; i < m_row; ++i)
	{
		for (size_t j = 0; j < m_col; ++j)
		{
			res.m_ptr[j][i] = this->m_ptr[i][j];
		}
	}
	return res;
}

Vector operator*(const Matrix& a, const Vector& b)
{
	if (a.cols() != b.size())
		throw std::runtime_error("Error Matrix and Vector multiplication\n");
	Vector res(a.rows());
	for (size_t i = 0; i < res.size(); ++i) {
		double tmp = 0;
		for (size_t k = 0; k < b.size(); ++k) {
			tmp += a(i,k) * b(k);
		}
		res(i) = tmp;
	}
	return res;
}
