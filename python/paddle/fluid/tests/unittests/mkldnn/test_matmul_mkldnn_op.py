# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest, os
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci


@skip_check_grad_ci(reason="DNNL's MatMul doesn't implemend grad kernel.")
class TestDnnlMatMulOp(OpTest):
    def generate_data(self):
        self.x = np.random.random((25, 2, 2)).astype("float32")
        self.y = np.random.random((25, 2, 2)).astype("float32")
        self.alpha = 1.0
        self.out = self.alpha * np.matmul(self.x, self.y)

    def set_attributes(self):
        self.alpha = self.alpha if hasattr(self, 'alpha') else 1.0
        self.attrs = {'alpha': self.alpha}

    def setUp(self):
        # Set max isa, otherwise fails on SKX and earlier
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()
        self.attrs['use_mkldnn'] = True

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


class TestDnnlMatMulOpAlpha(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.random((17, 2, 3)).astype("float32")
        self.y = np.random.random((17, 3, 2)).astype("float32")
        self.alpha = 2.0
        self.out = self.alpha * np.matmul(self.x, self.y)


class TestDnnlMatMulOp2D(TestDnnlMatMulOp):
    def print_tensor(self, name, tensor):
        print(name)
        print(tensor)

    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((9, 12)).astype("float32")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpTransposeX(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((12, 9)).astype("float32")
        self.out = np.matmul(np.transpose(self.x), self.y)

    def set_attributes(self):
        self.attrs = {'transpose_X': True}


class TestDnnlMatMulOpTransposeY(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((12, 9)).astype("float32")
        self.out = np.matmul(self.x, np.transpose(self.y))

    def set_attributes(self):
        self.attrs = {'transpose_Y': True}


class TestDnnlMatMulOpTransposeY3D(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.random((17, 3, 2)).astype("float32")
        self.y = np.random.random((17, 3, 2)).astype("float32")
        self.out = np.matmul(self.x, np.transpose(self.y, (0, 2, 1)))

    def set_attributes(self):
        self.attrs = {'transpose_Y': True}


class TestDnnlMatMulOpInt8NoScales(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("int8")
        self.y = np.random.random((9, 12)).astype("int8")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpInt8(TestDnnlMatMulOp):
    def quantize(self, tensor):
        scale = 127. / np.abs(np.amax(tensor))
        quantized = np.round(scale * tensor).astype("int8")
        return scale, quantized

    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out_scale, self.out = self.quantize(out_float)

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'Scale_out': self.out_scale,
        }

    def test_check_output(self):
        int_atol = 1
        self.check_output(atol=int_atol)


class TestDnnlMatMulOpInt8ForceFP32(TestDnnlMatMulOpInt8):
    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out = out_float

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'force_fp32_output': True
        }


class TestDnnlMatMulOpInt8ForceFP32BasicScales(TestDnnlMatMulOp):
    def generate_data(self):
        self.x = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.y = np.random.randint(0, 3, (9, 12)).astype("int8")
        self.out = np.matmul(self.x, self.y).astype("float32")

    def set_attributes(self):
        self.attrs = {'force_fp32_output': True}


@skip_check_grad_ci(reason="DNNL's MatMul doesn't implement grad kernel.")
class TestMatMulOpReshapeTranspose(OpTest):
    def init_data_type(self):
        self.data_type_ = 'float32'

    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
        self.y = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
        self.out = np.matmul(self.x, self.y.transpose([0, 1, 3, 2]))
        self.fused_reshape_X = []
        self.fused_transpose_X = []
        self.fused_reshape_Y = []
        self.fused_transpose_Y = []

    def setUp(self):
        # Set max isa, otherwise fails on SKX and earlier
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.transpose_y = True
        self.init_data_type(self)
        self.generate_data()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
            'transpose_Y': self.transpose_y
        }
        if len(self.fused_transpose_X) > 0:
            self.attrs['fused_transpose_X'] = self.fused_transpose_X
        if len(self.fused_transpose_Y) > 0:
            self.attrs['fused_transpose_Y'] = self.fused_transpose_Y
        if len(self.fused_reshape_X) > 0:
            self.attrs['fused_reshape_X'] = self.fused_reshape_X
        if len(self.fused_reshape_Y) > 0:
            self.attrs['fused_reshape_Y'] = self.fused_reshape_Y

        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


class TestMatMulOpReshapeTranspose4DXFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32")
        self.y = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
        self.fused_transpose_X = [0, 2, 1, 3]
        self.fused_reshape_X = [0, 0, 12, 64]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
            self.x.reshape([2, 128, 12, 64]).transpose([0, 2, 1, 3]),
            self.y.transpose([0, 1, 3, 2]))


class TestMatMulOpReshapeTranspose4DXInt8(TestMatMulOpReshapeTranspose4DXFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose4DYFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
        self.y = np.random.random([2, 128, 768]).astype("float32")
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [0, 2, 1, 3]
        self.fused_reshape_Y = [0, 0, 12, 64]
        self.out = np.matmul(
            self.x, self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1]))


class TestMatMulOpReshapeTranspose4DYInt8(TestMatMulOpReshapeTranspose4DYFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose4DXYFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32")
        self.y = np.random.random([2, 128, 768]).astype("float32")
        self.fused_transpose_X = [0, 2, 1, 3]
        self.fused_reshape_X = [0, 0, 12, 64]
        self.fused_transpose_Y = [0, 2, 1, 3]
        self.fused_reshape_Y = [0, 0, 12, 64]
        self.out = np.matmul(
            self.x.reshape([2, 128, 12, 64]).transpose([0, 2, 1, 3]),
            self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1]))


class TestMatMulOpReshapeTranspose4DXYInt8(
        TestMatMulOpReshapeTranspose4DXYFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose2DXFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 5, 10]).astype("float32")
        self.y = np.random.random([2, 5, 10]).astype("float32").reshape(
            [10, 10]).transpose([1, 0])
        self.fused_transpose_X = [1, 0]
        self.fused_reshape_X = [10, 10]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
            self.x.reshape([10, 10]).transpose([1, 0]),
            self.y.transpose([1, 0]))


class TestMatMulOpReshapeTranspose2DXInt8(TestMatMulOpReshapeTranspose2DXFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose2DYFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 5, 10]).astype("float32").reshape(
            [10, 10]).transpose([1, 0])
        self.y = np.random.random([2, 5, 10]).astype("float32")
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [1, 0]
        self.fused_reshape_Y = [10, 10]
        self.out = np.matmul(self.x, self.y.reshape([10, 10]))


class TestMatMulOpReshapeTranspose2DYInt8(TestMatMulOpReshapeTranspose2DYFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose3DXFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 2, 5, 5]).astype("float32")
        self.y = np.random.random([2, 2, 5, 5]).astype("float32").reshape(
            [2, 10, 5]).transpose([0, 2, 1])
        self.fused_transpose_X = [0, 2, 1]
        self.fused_reshape_X = [2, 10, 5]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
            self.x.reshape([2, 10, 5]).transpose(0, 2, 1),
            self.y.transpose(0, 2, 1))


class TestMatMulOpReshapeTranspose3DXInt8(TestMatMulOpReshapeTranspose3DXFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestMatMulOpReshapeTranspose3DYFloat(TestMatMulOpReshapeTranspose):
    def generate_data(self):
        self.x = np.random.random([2, 2, 5, 5]).astype(self.data_type_).reshape(
            [2, 10, 5]).transpose([0, 2, 1])
        self.y = np.random.random([2, 2, 5, 5]).astype(self.data_type_)
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [0, 2, 1]
        self.fused_reshape_Y = [2, 10, 5]
        self.out = np.matmul(self.x, self.y.reshape([2, 10, 5]))


class TestMatMulOpReshapeTranspose3DYInt8(TestMatMulOpReshapeTranspose3DYFloat):
    def init_data_type(self):
        self.data_type_ = 'int8'


if __name__ == "__main__":
    unittest.main()
