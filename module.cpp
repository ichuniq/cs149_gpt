#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    int index = x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b;
    return tensor[index];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    int index = x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b;
    tensor[index] = val; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embecing Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emveced dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embecing Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    
    // For each Batch
    for (int b = 0; b < B; b++) {
        // For each Head
        for (int h = 0; h < H; h++) {

            // a. Multiply Q (N, d) with K^t (d, N), storing it in QK^t (N, N)
            //     QK^t[i][j] += Q[i][k] * K^t[k][j] => Q[i][k] * K[j][k]
            //     So we can iterate j then k to emulate transpose K
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float qkt_val = 0.0;
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d); // Q[b][h][i][k]
                        float k_val = fourDimRead(K, b, h, j, k, H, N, d); // K[b][h][j][k]
                        qkt_val += q_val * k_val;
                    }
                    twoDimWrite(QK_t, i, j, N, qkt_val);
                }
            }
            
            // b. Perform softmax for each row in QK^t (N, N)
            for (int i = 0; i < N; i++) {
                // Calculate sum of exponentials of row i
                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    QK_t[i * N + j] = std::exp(QK_t[i * N + j]);
                    sum_exp += QK_t[i * N + j];
                }
                // Normalize each element
                for (int j = 0; j < N; j++) {
                    QK_t[i * N + j] /= sum_exp;
                }
            }

            // c. Matrix multiply QK^t with V and store it into O
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float o_val = 0.0f;
                    for (int k = 0; k < N; k++) {
                        float qk_val = twoDimRead(QK_t, i, k, N); // QK_t[i][k]
                        float v_val = fourDimRead(V, b, h, k, j, H, N, d); // V[b][h][k][j]
                        o_val += qk_val * v_val;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, o_val); // O[b][h][i][j]
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    // Define block size
    const int B_N = 32;
    const int B_d = 32;

    // For each Batch
    for (int b = 0; b < B; b++) {
        // For each Head
        for (int h = 0; h < H; h++) {

            // a. Blocked matmul of Q with K^t
            for (int i0 = 0; i0 < N; i0 += B_N) {
                for (int j0 = 0; j0 < N; j0 += B_d) {
                    // Ending indices of the current block
                    int i_max = std::min(i0 + B_N, N);
                    int j_max = std::min(j0 + B_d, N);

                    for (int k0 = 0; k0 < d; k0 += B_d) {
                        int k_max = std::min(k0 + B_d, d);

                        // Compute matmul for the current block
                        for (int i = i0; i < i_max; i++) {
                            for (int j = j0; j < j_max; j++) {
                                float qkt_val = QK_t[i * N + j];
                                for (int k = k0; k < k_max; k++) {
                                    float q_val = fourDimRead(Q, b, h, i, k, H, N, d); // Q[b][h][i][k]
                                    float k_val = fourDimRead(K, b, h, j, k, H, N, d); // K[b][h][j][k]
                                    qkt_val += q_val * k_val;
                                }
                                QK_t[i * N + j] = qkt_val;
                            }
                        }
                    }
                }
            }

            // b. Perform softmax for each row in QK^t (N, N)
            for (int i = 0; i < N; i++) {
                // Calculate sum of exponentials of row i
                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    QK_t[i * N + j] = std::exp(QK_t[i * N + j]);
                    sum_exp += QK_t[i * N + j];
                }
                // Normalize each element
                for (int j = 0; j < N; j++) {
                    QK_t[i * N + j] /= sum_exp;
                }
            }

            // c. Blocked matmul of QK^t with V and store it into O
            for (int i0 = 0; i0 < N; i0 += B_N) {
                for (int j0 = 0; j0 < d; j0 += B_d) {
                    int i_max = std::min(i0 + B_N, N);
                    int j_max = std::min(j0 + B_d, d);

                    for (int k0 = 0; k0 < N; k0 += B_N) {
                        int k_max = std::min(k0 + B_N, N);

                        // Compute matmul for the current block
                        for (int i = i0; i < i_max; i++) {
                            for (int j = j0; j < j_max; j++) {
                                float o_val = fourDimRead(O, b, h, i, j, H, N, d);
                                for (int k = k0; k < k_max; k++) {
                                    float qk_val = twoDimRead(QK_t, i, k, N); // QK_t[i][k]
                                    float v_val = fourDimRead(V, b, h, k, j, H, N, d); // V[b][h][k][j]
                                    o_val += qk_val * v_val;
                                }
                                fourDimWrite(O, b, h, i, j, H, N, d, o_val);
                            }
                        }
                    }
                }
            }

        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {

        // Loop over heads
        for (int h = 0; h < H; h++) {
            
            // For each row
            for (int i = 0; i < N ; i++) {
                
                // Each OpenMP thread gets its own copy
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);

                float sum_exp = 0.0f;
                // Compute Q[i] x K^T for all j
                for (int j = 0; j < N; j++) {
                    float score = 0.0;
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d); // Q[b][h][i][k]
                        float k_val = fourDimRead(K, b, h, j, k, H, N, d); // K[b][h][j][k]
                        score += q_val * k_val;
                    }
                    // Calculate exp. score for softmax along the way
                    ORow[j] = std::exp(score);
                    sum_exp += ORow[j];
                }
                // Normalize
                for (int j = 0; j < N; j++) {
                    ORow[j] /= sum_exp;
                }
                
                // Matmul of ORow (1, N) with V (N, d)
                // For each col of V
                for (int j = 0; j < d; j++) {
                    float o_val = 0.0f;
                    for (int k = 0; k < N; k++) {
                        float v_val = fourDimRead(V, b, h, k, j, H, N, d); // V[b][h][k][j]
                        o_val += ORow[k] * v_val;
                    }
                    // Write to O
                    fourDimWrite(O, b, h, i, j, H, N, d, o_val); // O[b][h][i][j]
                }
            }
	    }
    }
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        // Loop over heads
        for (int h = 0; h < H; h++) {

            // Loop over Tc = ceil(N / Bc) blocks
            for (int j0 = 0; j0 < N; j0 += Bc) {
                int j_max = std::min(j0 + Bc, N);

                // Load Kj and Vj, both have shape (Bc, d), to local memory blocks
                for (int c = j0; c < j_max; c++) {
                    for (int r = 0; r < d; r++) {
                        Kj[(c - j0)*d + r] = fourDimRead(K, b, h, c, r, H, N, d); // K[b][h][c][r]
                        Vj[(c - j0)*d + r] = fourDimRead(V, b, h, c, r, H, N, d); // V[b][h][c][r]
                    }
                }

                // Loop over Tr = ceil(N / Br) blocks
                for (int i0 = 0; i0 < N; i0 += Br) {
                    int i_max = std::min(i0 + Br, N);

                    // Load Q_i, O_i and l_i to local memory
                    // Q_i, O_i, PV have shape (Br, d); l_i, l_ij, l_new have shape (Br)
                    for (int r = i0; r < i_max; r++) {
                        for (int c = 0; c < d; c++) {
                            Qi[(r - i0) * d + c] = fourDimRead(Q, b, h, r, c, H, N, d);
                            Oi[(r - i0) * d + c] = fourDimRead(O, b, h, r, c, H, N, d);
                        }
                        li[r - i0] = l[r];
                    }

                    // Compute S_ij (Br, Bc) <-- Q_i (Br, d) @ K_j^T (d, Bc)
                    // Compute P_ij (Br, Bc) <-- exp(S_ij)
                    for (int r = i0; r < i_max; r++) {
                        int rr = r - i0; // local row index

                        for (int c = j0; c < j_max; c++) {
                            int cc = c - j0; // local column index

                            float qkt_val = Sij[rr * Bc + cc];
                            for (int k = 0; k < d; k++) {
                                float q_val = Qi[rr * d + k]; // Qi[r - i0][k];
                                float k_val = Kj[cc * d + k]; // Kj[c - j0][k];
                                qkt_val += q_val * k_val;
                            }
                            Sij[rr * Bc + cc] = qkt_val;
                            Pij[rr * Bc + cc] = std::exp(qkt_val);
                        }
                    }

                    // Compute l_ij (Br) <-- rowSum(P_ij)
                    for (int r = i0; r < i_max; r++) {
                        int rr = r - i0; // local row index
                        float row_sum = 0.0;
                        for (int c = j0; c < j_max; c++) {
                            int cc = c - j0; // local column index
                            row_sum += Pij[rr * Bc + cc];
                        }
                        lij[rr] = row_sum;
                        // l_new = l_i + l_ij
                        lnew[rr] = li[rr] + lij[rr];
                    }
                    
                    // Compute O_i (Br, d)  <--  (l_i O_i + P_ij @ V_j) / l_new
                    // s.t. P_ij and V_j is matmul, l_i and O_i is elementwise mul.


                    // Write blocks O_i and l_new back to O and l in main memory

                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
