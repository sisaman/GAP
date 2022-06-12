#include <torch/script.h>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;
using namespace torch::indexing;

tuple<torch::Tensor, torch::Tensor> 
sample_edge(const vector<int64_t>& row, const vector<int64_t>& col, const int64_t num_nodes, const int64_t max_deg)
{
  srand(time(0));
  int num_edges = row.size();
  vector<int64_t> deg(num_nodes, 0);
  vector<int64_t> row_sampled; row_sampled.reserve(num_nodes * max_deg);
  vector<int64_t> col_sampled; col_sampled.reserve(num_nodes * max_deg);
  
  for (int i = 0; i < num_edges; ++i)
  {
    int u = row[i];
    int v = col[i];

    if (deg[u] < max_deg && deg[v] < max_deg)
    {
      row_sampled.push_back(u);
      col_sampled.push_back(v);
      ++deg[v];
      ++deg[u];
    }
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor row_out = torch::from_blob(row_sampled.data(), {row_sampled.size()}, opts);
  torch::Tensor col_out = torch::from_blob(col_sampled.data(), {col_sampled.size()}, opts);
  return make_pair(row_out.clone(), col_out.clone());
}


TORCH_LIBRARY(my_ops, m)
{
  m.def("sample_edge", &sample_edge);
}

