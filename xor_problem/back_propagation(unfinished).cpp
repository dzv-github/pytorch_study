#include <iostream>
#include<algorithm>
#include<math.h>
#include<cstdlib>
#include<time.h>
#include<map>
#include<random>
#include<vector>

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
		else if (st_column != var.st_column) {
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
struct bias_location {
	int prior_layer;
	int next_layer;

	bool operator <(const bias_location& var) const
	{
		if (prior_layer != var.prior_layer) {
			return prior_layer < var.prior_layer;
		}
		else {
			return next_layer < var.next_layer;
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

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double>dis(0, 1);

map<weight_location, long double>weight;
map<weight_location, long double>new_weight;
map<bias_location, long double>bias;
map<node_location, vector<node_location> >prior_nodes;
map<node_location, vector<node_location> >next_nodes;
map<node_location, node_values>node; //pair<net,out>
map<node_location, long double>delta;
vector<int>network_size;
vector<string>act_func;
vector<bool>bias_usage;
long double temp_weight=0.15;

int epoch = 10000;
int input_size = 1;
long double learning_rate = 0.5;

long double input[5][3] = { {0.05, 0.1} };
long double correct_output[5][3] = {{0.01,0.99}};
long double output[5][3];
map<int,long double> error;
long double total_error;

void connect() {
	for (int i = 0;i < network_size.size();i++) {
		for (int j = 0;j < network_size[i];j++) {

			if (i != network_size.size() - 1) {
				for (int k = 0;k < network_size[i + 1];k++) {
					next_nodes[{i, j}].push_back({ i + 1,k });
					weight[{i, j, i + 1, k}] = temp_weight;
					temp_weight += 0.05;
				}
				
			}

			if (i != 0) {
				for (int k = 0;k < network_size[i - 1];k++) {
					prior_nodes[{i, j}].push_back({ i - 1,k });
				}
			}
		}

		bias[{i, i + 1}] = temp_weight;
		temp_weight += 0.05;
	}

	return;
}



long double ReLU(long double x) {
	return max(0, x);
}
long double sigmoid(long double x) {
	return 1 / (1 + exp(-x));
}


void perceptron(int layer, int column, string activative_func, bool set_bias) {
	long double result = 0;
	if (set_bias == true)result = bias[{layer-1, layer}];

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

	linear(2, true);
	act_func.push_back("sigmoid");

	linear(2,true);
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
map<weight_location, long double> temp1, temp2, temp3, temp4;
int size_of_network;
void backward(string loss_fuction,int input_num) {

	size_of_network = network_size.size()-1;
	for (int j = 0;j < network_size[size_of_network];j++) {
		for (int k = 0;k < prior_nodes[{size_of_network,j}].size();k++) {
			temp1[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] = 
				-(correct_output[input_num][j] - node[{size_of_network, j}].out);

			temp2[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] =
				node[{size_of_network, j}].out* (1 - node[{size_of_network, j}].out);

			temp3[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] =
				node[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column}].out;

			temp4[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] =
				temp1[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] * 
				temp2[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] *
				temp3[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}];

			new_weight[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] =
				weight[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}] - 
				(learning_rate * temp4[{prior_nodes[{size_of_network, j}][k].layer, prior_nodes[{size_of_network, j}][k].column, size_of_network, j}]);
		}
	}

	for (int i = size_of_network - 1;i >= 0;i--) {
		for (int j = 0;j < network_size[i];j++) {
			for (int k = 0;k < prior_nodes[{i, j}].size();k++) {
				temp1[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] = 0;
				for (int n = 0;n < next_nodes[{i, j}].size();n++) {
					temp1[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] +=
						(temp1[{i, j, next_nodes[{i, j}][n].layer, next_nodes[{i, j}][n].column}] *
							temp2[{i, j, next_nodes[{i, j}][n].layer, next_nodes[{i, j}][n].column}])*
						weight[{i, j, next_nodes[{i, j}][n].layer, next_nodes[{i, j}][n].column}];

				}

				temp2[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] = node[{i, j}].out* (1 - node[{i, j}].out);
				temp3[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] = node[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column}].out;
				temp4[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] =
					temp1[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] *
					temp2[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] *
					temp3[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}];

				new_weight[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] = weight[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}] -
					learning_rate * temp4[{prior_nodes[{i, j}][k].layer, prior_nodes[{i, j}][k].column, i, j}];
			}
		}
	}
	weight = new_weight;
	return;
}
int main()
{
	sequential();

	for (int step = 0;step < epoch;step++) {
		total_error = 0;
		for (int i = 0;i < input_size;i++) {
			forward_Pass(i);
			output[i][0] = node[{size_of_network, 0}].out;
			output[i][1] = node[{size_of_network, 1}].out;
			error[0] = pow((output[i][0] - correct_output[i][0]), 2) / 2;
			error[1] = pow((output[i][1] - correct_output[i][1]), 2) / 2;
			cout << output[i][0] << " " << output[i][1]<<endl;
			total_error = pow((output[i][0] - correct_output[i][0]), 2) / 2 + pow((output[i][1] - correct_output[i][1]), 2) / 2;
			cout << total_error << endl;
			backward("MSELoss",i);

		}
	}
	return 0;
}
