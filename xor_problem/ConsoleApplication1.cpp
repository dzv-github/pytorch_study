
#include <iostream>
#include<algorithm>
#include<math.h>
#include<cstdlib>
#include<time.h>
#include<map>
#include<vector>
#define M_E 2.718

using namespace std;

struct weight_location
{
	int st_layer;
	int st_column;
	int ed_layer;
	int ed_column;

	bool operator <(const weight_location& var) const
	{
		if (st_layer != var.st_layer) {
			return st_layer < var.st_layer;
		}
		else if(st_column!=var.st_column){
			return st_column < var.st_column;
		}
		else if (ed_layer != var.ed_layer) {
			return ed_layer < var.ed_layer;
		}
		else {
			return ed_column < var.ed_column;
		}
	}
};
struct node_location {
	int layer;
	int column;

	bool operator <(const node_location& var) const
	{
		if (layer != var.layer) {
			return layer < var.layer;
		}
		else {
			return column < var.column;
		}
	}
};
struct node_values {
	long double net;
	long double out;
};

long double max(long double a, long double b) {
	if (a > b)return a;
	else return b;
}

map<weight_location, long double>weight;
map<node_location, long double>bias;
map<node_location, vector<node_location>>prior_nodes;
map<node_location, vector<node_location>>next_nodes;
map<node_location, node_values>node; //pair<net,out>
map<node_location, long double>delta;
vector<int>network_size;
vector<string>act_func;
vector<bool>bias_usage;

int epoch=10000;
int input_size = 4;
long double learning_rate = 0.1;

long double input[5][3] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
long double correct_output[5] = { 0,1,1,0 };
long double output[5];


void connect() {
	for (int i = 0;i < network_size.size();i++) {
		for (int j = 0;j < network_size[i];j++) {

			if (i != network_size.size() - 1) {
				for (int k = 0;k < network_size[i + 1];k++) {
					next_nodes[{i, j}].push_back({ i + 1,k });
					weight[{i, j, i + 1, k}] = abs(rand()%10)*0.1;
				}
			}

			if (i != 0) {
				for (int k = 0;k < network_size[i - 1];k++) {
					prior_nodes[{i, j}].push_back({ i - 1,k });
				}
			}
		}
	}
	return;
}

long double ReLU(long double x) {
	return max(0, x);
}
long double sigmoid(long double x) {
	return 1 / (1 + exp(-x));
}

void perceptron(int layer, int column,string activative_func,bool set_bias) {
	long double result = 0;
	if (set_bias == true)result = bias[{layer, column}];

	for (int i = 0;i < prior_nodes[{layer, column}].size();i++) {
		result += weight[{ prior_nodes[{layer, column}][i].layer, prior_nodes[{layer, column}][i].column, layer, column}] *
			node[{prior_nodes[{layer, column}][i].layer, prior_nodes[{layer, column}][i].column}].out;
	}

	if (activative_func == "sigmoid")
		node[{layer, column}] = { result,sigmoid(result) };
	else if (activative_func == "ReLU")
		node[{layer, column}] = { result,ReLU(result) };
	else
		node[{layer, column}] = { result,result };
	return;
}

void linear(int net_size, bool biasUsage) {
	network_size.push_back(net_size);
	bias_usage.push_back(biasUsage);
}

void sequential() {
	linear(2, true);
	act_func.push_back("sigmoid");

	linear(10, true);
	act_func.push_back("sigmoid");

	linear(10, true);
	act_func.push_back("sigmoid");

	linear(1, true);
	act_func.push_back("sigmoid");

	connect();
	return;
}

void forward_Pass(int n) {
	for (int i = 0;i < network_size[0];i++) {
		node[{0, i}] = { input[n][i],input[n][i] };
	}

	for (int i = 1;i < network_size.size();i++) {
		for (int j = 0;j < network_size[i];j++) {
			perceptron(i, j, act_func[i], bias_usage[i]);
		}
	}
	return;
}

void delta_init() {
}

void backward() {
}
int main()
{
	srand(time(NULL));

	sequential();

	for (int step = 0;step < epoch;step++) {
		for (int i = 0;i < input_size;i++) {
			forward_Pass(i);
			output[i] = node[{input_size-1, 0}].out;
		}
		
		backward();
		delta_init();
		printf("%f %f %f %f\n", output[0], output[1], output[2], output[3]);
	}

	printf("%f", node[{3, 0}].out);
	return 0;
}
