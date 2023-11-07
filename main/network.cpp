#include "network.h"

Network::Network(size_t layers, const std::vector<size_t>& neurons_size, ActivateFunction::FunctionType func_type)
{
	m_layers = layers;
	m_neurons_size = neurons_size;
	m_act_func.set(func_type);

	for (size_t i = 0; i < m_layers - 1; ++i)
	{
		//Создаем матрицы весов
		m_weights.emplace_back(m_neurons_size[i + 1], m_neurons_size[i]);
		m_weights.back().rand();

		//Создаем матрицы смещений
		m_offset.emplace_back(m_neurons_size[i + 1]);
		m_offset.back().rand();

		//Задаем нейроны смещения
		m_neurons_offset.push_back(1.0);
	}

	for (size_t i = 0; i < m_layers; ++i)
	{
		m_neurons.emplace_back(m_neurons_size[i]);
		m_neurons_error.emplace_back(m_neurons_size[i]);
	}
}

void Network::input(const std::vector<double>& values)
{
	for (size_t i = 0; i < m_neurons_size[0]; ++i)
		m_neurons[0](i) = values[i];
}

void Network::forwardFeed()
{
	for (size_t i = 1; i < m_layers; ++i)
	{
		m_neurons[i] = m_weights[i - 1] * m_neurons[i - 1];
		m_neurons[i] += m_offset[i - 1];
		m_act_func.use(m_neurons[i]);
	}
}

size_t Network::prediction()
{
	double max = m_neurons[m_layers - 1](0);
	size_t pred = 0;
	double tmp;
	for (size_t i = 1; i < m_neurons_size[m_layers - 1]; ++i)
	{
		tmp = m_neurons[m_layers - 1](i);
		if (tmp > max) {
			pred = i;
			max = tmp;
		}
	}
	return pred;
}

void Network::backPropogation(double expect) {
	for (size_t i = 0; i < m_neurons_size[m_layers - 1]; ++i) {
		if (i != static_cast<int>(expect))
			m_neurons_error[m_layers - 1](i) = -m_neurons[m_layers - 1](i) * m_act_func.useDerivative(m_neurons[m_layers - 1](i));
		else
			m_neurons_error[m_layers - 1](i) = (1.0 - m_neurons[m_layers - 1](i)) * m_act_func.useDerivative(m_neurons[m_layers - 1](i));
	}
	for (size_t k = m_layers - 2; k > 0; --k) {
		m_neurons_error[k] = m_weights[k].trans() * m_neurons_error[k + 1];
		for (size_t j = 0; j < m_neurons_size[k]; ++j)
			m_neurons_error[k](j) *= m_act_func.useDerivative(m_neurons[k](j));
	}
}
void Network::updateWeights(double lr) {
	for (size_t i = 0; i < m_layers - 1; ++i) {
		for (size_t j = 0; j < m_neurons_size[i + 1]; ++j) {
			for (size_t k = 0; k < m_neurons_size[i]; ++k) {
				m_weights[i](j, k) += m_neurons[i](k) * m_neurons_error[i + 1](j) * lr;
			}
		}
	}
	for (size_t i = 0; i < m_layers - 1; ++i) {
		for (size_t k = 0; k < m_neurons_size[i + 1]; ++k) {
			m_offset[i](k) += m_neurons_error[i + 1](k) * lr;
		}
	}
}
