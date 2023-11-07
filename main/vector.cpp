#include "matrix.h"
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

Vector::Vector(size_t size)
{
	m_size = size;
	m_ptr = new double[m_size];
}

Vector::Vector(size_t size, double value) : Vector(size)
{
	for (size_t i = 0; i < m_size; ++i)
		m_ptr[i] = value;
}
Vector::Vector(const Vector& other) : Vector(other.m_size)
{
	std::memcpy(m_ptr, other.m_ptr, m_size * sizeof(double));
}
Vector::Vector(Vector&& other) noexcept : m_size(other.m_size), m_ptr(other.m_ptr)
{
	other.m_ptr = nullptr;
	other.m_size = 0;
}
Vector::~Vector()
{
	delete[] m_ptr;
}
double& Vector::at(size_t i)
{
	if (i >= m_size)
		throw std::out_of_range("Out of range in Vector");
	return m_ptr[i];
}
double& Vector::operator()(size_t i) noexcept
{
	return m_ptr[i];
}
double Vector::at(size_t i) const
{
	if (i >= m_size)
		throw std::out_of_range("Out of range in Vector");
	return m_ptr[i];
}
double Vector::operator()(size_t i) const noexcept
{
	return m_ptr[i];
}
size_t Vector::size() const
{
	return m_size;
}
void Vector::rand()
{
	std::srand(std::time(nullptr));
	for (size_t i = 0; i < m_size; ++i)
		m_ptr[i] = (std::rand() % 1000) * 0.017 / (m_size + 42);
}

Vector& Vector::operator+=(const Vector& b)
{
	if (m_size != b.m_size)
		throw std::runtime_error("Error Vector add\n");
	for (size_t i = 0; i < m_size; ++i) {
		m_ptr[i] += b.m_ptr[i];
	}
	return *this;
}

Vector& Vector::operator=(const Vector& b)
{
	delete[] m_ptr;

	m_size = b.m_size;
	m_ptr = new double[m_size];

	std::memcpy(m_ptr, b.m_ptr, m_size * sizeof(double));

	return *this;
}

Vector Vector::operator+(const Vector& b) const
{
	Vector sum = *this;
	sum += b;
	return sum;
}