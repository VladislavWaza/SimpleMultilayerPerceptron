#pragma once
#include "activateFunction.h"
#include <vector>
class Network
{
	size_t m_layers; //����� �����
	std::vector<size_t> m_neurons_size; //����� ��������� �� �����
	ActivateFunction m_act_func;
	std::vector<Matrix> m_weights; //������� �����
	std::vector<Vector> m_offset; //���� ��������
	std::vector<Vector> m_neurons; //�������� ��������
	std::vector<Vector> m_neurons_error; //������ ��������
	std::vector<double> m_neurons_offset; //������� ��������
public:
	Network(size_t layers, const std::vector<size_t>& neurons_size, ActivateFunction::FunctionType func_type);
	void input(const std::vector<double>& values);
	void forwardFeed();
	size_t prediction();
	void backPropogation(double expect);
	void updateWeights(double lr);
};

