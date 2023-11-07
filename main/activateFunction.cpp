#include "activateFunction.h"
#include <stdexcept>

void ActivateFunction::set(FunctionType func_type)
{
	m_func = func_type;
}

void ActivateFunction::use(Vector& v)
{
	size_t size = v.size();
	switch (m_func)
	{
	case ActivateFunction::FunctionType::Sigmoid:
		for (size_t i = 0; i < size; ++i)
			v(i) = 1 / (1 + exp(-v(i)));
		break;
	case ActivateFunction::FunctionType::ReLU:
		for (size_t i = 0; i < size; ++i) {
			if (v(i) < 0)
				v(i) *= 0.01;
			else if (v(i) > 1)
				v(i) =  1.0 + 0.01 * (v(i) - 1.0);
		}
		break;
	case ActivateFunction::FunctionType::Tanh:
		for (size_t i = 0; i < size; ++i) {
			if (v(i) < 0)
				v(i) = 0.01 * (exp(v(i)) - exp(-v(i))) / (exp(v(i)) + exp(-v(i)));
			else
				v(i) = (exp(v(i)) - exp(-v(i))) / (exp(v(i)) + exp(-v(i)));
		}
		break;
	default:
		throw std::runtime_error("Error use ActivateFunction");
		break;
	}
}

void ActivateFunction::useDerivative(Vector& v)
{
	size_t size = v.size();
	switch (m_func)
	{
	case ActivateFunction::FunctionType::Sigmoid:
		for (size_t i = 0; i < size; ++i)
			v(i) = v(i) * (1 - v(i));
		break;
	case ActivateFunction::FunctionType::ReLU:
		for (size_t i = 0; i < size; ++i) {
			if (v(i) < 0 || v(i) > 1)
				v(i) = 0.01;
			else
				v(i) = 1;
		}
		break;
	case ActivateFunction::FunctionType::Tanh:
		for (size_t i = 0; i < size; ++i) {
			if (v(i) < 0)
				v(i) = 0.01 * (1 - v(i) * v(i));
			else
				v(i) = 1 - v(i) * v(i);
		}
		break;
	default:
		throw std::runtime_error("Error useDerivative ActivateFunction");
		break;
	}
}

double ActivateFunction::useDerivative(double x)
{
	double y = 0.0;
	switch (m_func)
	{
	case ActivateFunction::FunctionType::Sigmoid:
		y = x * (1 - x);
		break;
	case ActivateFunction::FunctionType::ReLU:
		if (x < 0 || x > 1)
			y = 0.01;
		else
			y = 1;
		break;
	case ActivateFunction::FunctionType::Tanh:
		if (x < 0)
			y = 0.01 * (1 - x * x);
		else
			y = 1 - x * x;
		break;
	default:
		throw std::runtime_error("Error useDerivative ActivateFunction");
		break;
	}
	return y;
}
