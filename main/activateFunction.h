#pragma once
#include "matrix.h"

class ActivateFunction
{
public:
	enum class FunctionType { Sigmoid, ReLU, Tanh};
private:
	FunctionType m_func = FunctionType::Sigmoid;
public:
	void set(FunctionType func_type);
	void use(Vector& v);
	void useDerivative(Vector& v);
	double useDerivative(double x);
};

