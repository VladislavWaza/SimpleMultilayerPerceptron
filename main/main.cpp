#include <iostream>
#include <fstream>
#include <chrono>
#include "matrix.h"
#include "activateFunction.h"
#include "network.h"

struct ExampleData {
	int answer = 0;
	std::vector<double> pixels;
};

void ReadData(std::string path, size_t imgSize, size_t& examples, std::vector<ExampleData>& data);

int main()
{

	const ActivateFunction::FunctionType af = ActivateFunction::FunctionType::ReLU;
	const size_t l = 3;
	const std::vector<size_t> sizes = { 784, 256, 10 };
	Network NW(l,sizes, af);

	std::cout << l << " layers\n";
	std::cout << "Sizes of layers: ";
	for (size_t i = 0; i < l; ++i)
		std::cout << sizes[i] << " ";
	std::cout << '\n';
	if (af == ActivateFunction::FunctionType::Sigmoid)
		std::cout << "Activate function: sigmoid\n";
	else if (af == ActivateFunction::FunctionType::ReLU)
		std::cout << "Activate function: ReLU\n";
	else if (af == ActivateFunction::FunctionType::Tanh)
		std::cout << "Activate function: Tanh\n";




	size_t examples = 0;
	std::vector<ExampleData> data;
	ReadData("lib_MNIST_edit.txt", sizes[0], examples, data);
	auto begin = std::chrono::steady_clock::now();

	int score = 0, bestScore = 0;
	int epoch = 0; 
	const int maxEpoch = 10;
	while (score / examples * 100 < 100) {
		score = 0;
		auto t1 = std::chrono::steady_clock::now();
		for (int i = 0; i < examples; ++i) {
			NW.input(data[i].pixels);
			NW.forwardFeed();
			int pred = NW.prediction();
			if (pred != data[i].answer) {
				NW.backPropogation(data[i].answer);
				NW.updateWeights(0.15 * exp(-epoch / static_cast<double>(maxEpoch)));
			}
			else {
				++score;
			}
		}
		auto t2 = std::chrono::steady_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
		if (score > bestScore) 
			bestScore = score;
		std::cout << "score: " << score / static_cast<double>(examples) * 100 << "\t" 
			      << "bestScore: " << bestScore / static_cast<double>(examples) * 100 << "\t" 
				  << "epoch: " << epoch << "\tTIME: " << time.count() << std::endl;
		++epoch;
		if (epoch == maxEpoch)
			break;
	}

	{
		auto end = std::chrono::steady_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
		std::cout << "TIME: " << time.count() / 60. << " min" << std::endl;
	}

	{
		size_t tests = 0;
		std::vector<ExampleData> test_data;
		ReadData("lib_10k.txt", sizes[0], tests, test_data);
		int score = 0;
		for (int i = 0; i < tests; ++i) {
			NW.input(test_data[i].pixels);
			NW.forwardFeed();
			int pred = NW.prediction();
			if (pred == test_data[i].answer)
				++score;
		}
		std::cout << "Score: " << score / static_cast<double>(tests) * 100 << std::endl;
	}
}

void ReadData(std::string path, size_t imgSize, size_t& examples, std::vector<ExampleData>& data)
{
	std::ifstream in;
	in.open(path);
	if (!in.is_open()) {
		std::cout << "Error open file " << path << '\n';
		system("pause");
	}
	else {
		std::string tmp;
		in >> tmp;
		if (tmp == "Examples") {
			in >> examples;
			std::cout << "Examples: " << examples << std::endl;
			data.resize(examples);
			for (size_t i = 0; i < examples; ++i) {
				in >> data[i].answer;
				data[i].pixels.resize(imgSize);
				for (size_t j = 0; j < imgSize; ++j) {
					in >> data[i].pixels[j];
				}
			}
			std::cout << path << " loaded\n";
		}
		else {
			std::cout << path << "not loaded\n";
		}
		in.close();
	}
}