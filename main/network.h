#pragma once
#include "activateFunction.h"
#include <vector>
class Network
{
	size_t m_layers; //число слоев
	std::vector<size_t> m_neurons_size; //число нейтровов на слоях
	ActivateFunction m_act_func;
	std::vector<Matrix> m_weights; //матрицы весов
	std::vector<Vector> m_offset; //веса смещения
	std::vector<Vector> m_neurons; //значения нейронов
	std::vector<Vector> m_neurons_error; //ошибки нейронов
	std::vector<double> m_neurons_offset; //нейроны смещения
public:
	Network(size_t layers, const std::vector<size_t>& neurons_size, ActivateFunction::FunctionType func_type);
	void input(const std::vector<double>& values);
	void forwardFeed();
	size_t prediction();
	void backPropogation(double expect);
	void updateWeights(double lr);
};

