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
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
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

                     //loop over Embedding Dimensionality
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

            // a. Matmul of Q with K^t
            for (int i0 = 0; i0 < N; i0 += B_N) {
                for (int j0 = 0; j0 < N; j0 += B_d) {
                    int i_max = std::min(i0 + B_N, N);
                    int j_max = std::min(j0 + B_N, N);

                    // block for local QK_t result
                    float QK_t_block[B_N][B_N] = {0};

                    for (int k0 = 0; k0 < d; k0 += B_d) {
                        int k_max = std::min(k0 + B_d, d);

                        // load blocks of Q and K into local buffers
                        float Q_block[B_N][B_d] = {0};
                        float K_block[B_N][B_d] = {0};
                        for (int i = i0; i < i_max; i++) {
                            for (int k = k0; k < k_max; k++) {
                                Q_block[i - i0][k - k0] = fourDimRead(Q, b, h, i, k, H, N, d); // Q[b][h][i][k]
                            }
                        }
                        for (int j = j0; j < j_max; j++) {
                            for (int k = k0; k < k_max; k++) {
                                K_block[j - j0][k - k0] = fourDimRead(K, b, h, j, k, H, N, d); // K[b][h][j][k]
                            }
                        }

                        // Compute mat mul of the local block
                        for (int i = i0; i < i_max; i++) {
                            for (int j = j0; j < j_max; j++) {
                                for (int k = k0; k < k_max; k++) {
                                    float q_val = Q_block[i - i0][k - k0];
                                    float k_val = K_block[j - j0][k - k0];
                                    QK_t_block[i - i0][j - j0] += q_val * k_val;
                                }
                            }
                        }
                    }

                    // Write the computed block back to O
                    for (int i = i0; i < i_max; i++) {
                        for (int j = j0; j < j_max; j++) {
                            float qkt_val = QK_t_block[i - i0][j - j0];
                            twoDimWrite(QK_t, i, j, N, qkt_val);
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

                    // Allocate a local block for O
                    float o_block[B_N][B_d] = {0};

                    for (int k0 = 0; k0 < N; k0 += B_N) {
                        int k_max = std::min(k0 + B_N, N);

                        // Compute matmul of the local block
                        for (int i = i0; i < i_max; i++) {
                            for (int j = j0; j < j_max; j++) {
                                for (int k = k0; k < k_max; k++) {
                                    float qk_val = twoDimRead(QK_t, i, k, N); // QK_t[i][k]
                                    float v_val = fourDimRead(V, b, h, k, j, H, N, d); // V[b][h][k][j]
                                    o_block[i - i0][j - j0] += qk_val * v_val;
                                }
                            }
                        }
                    }

                    // Write the computed block back to O
                    for (int i = i0; i < i_max; i++) {
                        for (int j = j0; j < j_max; j++) {
                            float o_val = o_block[i - i0][j - j0];
                            fourDimWrite(O, b, h, i, j, H, N, d, o_val); // O[b][h][i][j]
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
    //loop over batch
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
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
