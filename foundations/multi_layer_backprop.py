import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        # Convert to arrays
        x_arr = np.array(x)
        W1_arr = np.array(W1)
        b1_arr = np.array(b1)
        W2_arr = np.array(W2)
        b2_arr = np.array(b2)
        y_true_arr = np.array(y_true)
        
        # 1. FORWARD PASS
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)
        y_pred = np.dot(W2_arr, a1) + b2_arr
        loss = np.mean((y_pred - y_true_arr) ** 2)

        # 2. BACKWARD PASS
        out_size = len(y_true)
        dL_dypred = (2.0 / out_size) * (y_pred - y_true_arr)
        
        db2 = dL_dypred
        dW2 = np.outer(dL_dypred, a1)
        
        dL_da1 = np.dot(W2_arr.T, dL_dypred)
        dL_dz1 = dL_da1 * (z1 > 0)
        
        db1 = dL_dz1
        dW1 = np.outer(dL_dz1, x_arr)
        
        # 3. ROUND AND PURGE SIGNED ZEROES (+ 0.0)
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': (np.round(dW1, 4) + 0.0).tolist(),
            'db1': (np.round(db1, 4) + 0.0).tolist(),
            'dW2': (np.round(dW2, 4) + 0.0).tolist(),
            'db2': (np.round(db2, 4) + 0.0).tolist()
        }