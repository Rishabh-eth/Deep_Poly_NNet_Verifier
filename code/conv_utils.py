import numpy as np
from scipy import linalg
import torch

def toeplitz_1_ch(kernel, input_size):
  # shapes
  k_h, k_w = kernel.shape
  i_h, i_w = input_size
  o_h, o_w = i_h - k_h + 1, i_w - k_w + 1

  # construct 1d conv toeplitz matrices for each row of the kernel
  toeplitz = []
  for r in range(k_h):
    toeplitz.append(linalg.toeplitz(c=(kernel[r, 0], *np.zeros(i_w - k_w)), r=(*kernel[r], *np.zeros(i_w - k_w))))

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
  h_blocks, w_blocks = o_h, i_h
  h_block, w_block = toeplitz[0].shape

  W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

  for i, B in enumerate(toeplitz):
    for j in range(o_h):
      W_conv[j, :, i + j, :] = B

  W_conv.shape = (h_blocks * h_block, w_blocks * w_block)
  return W_conv


def toeplitz_mult_ch(kernel, input_size, padding=1):
  """Compute toeplitz matrix for 2d conv with multiple in and out channels.
  Args:
      kernel: shape=(n_out, n_in, H_k, W_k)
      input_size: (n_in, H_i, W_i)"""
  r, m, n = input_size
  kernel_size = kernel.shape
  output_size = (kernel_size[0], input_size[1] - (kernel_size[2] - 1) + 2 * padding,
                 input_size[2] - (kernel_size[3] - 1) + 2 * padding)
  T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))
  pad = padding_matrix(m, n, padding=padding)
  for i, ks in enumerate(kernel):  # loop over output channel
    for j, k in enumerate(ks):  # loop over input channel
      T_k = toeplitz_1_ch(k, (m + 2 * padding, n + 2 * padding))
      T[i, :, j, :] = T_k @ pad

  T.shape = (np.prod(output_size), np.prod(input_size))

  return T


def padding_matrix(m, n, padding=1):
  "Returns matrix euqivalent for padding=1"
  p = padding
  x_off = p * (n + 2 * p) + p
  y_off = 0
  pad = np.zeros(((m + 2 * p) * (n + 2 * p), m * n))
  for i in range(m):
    pad[x_off:x_off + n, y_off:y_off + n] = np.eye(n, n)
    x_off += n + 2 * p
    y_off += n
  return pad


def toeplitz_mult_ch_with_stride(kernel, input_size, stride, padding=1):
  "Includes Zero padding=1"
  r, m, n = input_size
  t = kernel.shape[0]
  T = toeplitz_mult_ch(kernel, (r, m, n), padding=padding)

  m_out = m - (kernel.shape[-2] - 1) + 2 * padding
  n_out = n - (kernel.shape[-1] - 1) + 2 * padding

  col_selector = np.zeros(n_out)
  col_selector[::stride] = 1
  mask = np.zeros((m_out, n_out), dtype='float32')
  mask[::stride] = col_selector
  mask = np.reshape(mask, (-1))
  mask = np.tile(mask, t)
  T = T[mask > 0]
  return torch.from_numpy(T.astype('float32'))
