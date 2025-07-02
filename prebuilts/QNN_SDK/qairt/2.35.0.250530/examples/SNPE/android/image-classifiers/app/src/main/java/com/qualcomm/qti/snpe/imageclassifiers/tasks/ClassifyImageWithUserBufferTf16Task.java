//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Pair;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.TF16UserBufferTensor;
import com.qualcomm.qti.snpe.TensorAttributes;
import com.qualcomm.qti.snpe.UserBufferTensor;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ClassifyImageWithUserBufferTf16Task extends AbstractClassifyImageTask {

    private static final String LOG_TAG = ClassifyImageWithUserBufferTf16Task.class.getSimpleName();

    private static final int TF16_SIZE = 2;

    private static final int TF16_BITWIDTH = 16;

    private static final int mStepExactly0 = 0;

    private static final float mStepSize = 0.1f;

    public ClassifyImageWithUserBufferTf16Task(ModelOverviewFragmentController controller,
                                              NeuralNetwork network, Bitmap image, Model model) {
        super(controller, network, image, model);
    }

    @Override
    protected String[] doInBackground(Bitmap... params) {
        final List<String> result = new LinkedList<>();

        final Map<String, TF16UserBufferTensor> inputTensors = new HashMap<>();
        final Map<String, TF16UserBufferTensor> outputTensors = new HashMap<>();

        final Map<String, ByteBuffer> inputBuffers = new HashMap<>();
        final Map<String, ByteBuffer> outputBuffers = new HashMap<>();

        boolean status = prepareInputs(inputTensors, inputBuffers);
        if (!status) {
            return new String[0];
        }
        prepareOutputs(outputTensors, outputBuffers);

        final long javaExecuteStart = SystemClock.elapsedRealtime();
        status = mNeuralNetwork.execute(inputTensors, outputTensors);
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        if (!status) {
            return new String[0];
        }
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

        float[] outputValues = dequantize(outputTensors.get(mOutputLayer), outputBuffers.get(mOutputLayer));
        for (Pair<Integer, Float> pair : topK(1, outputValues)) {
            result.add(mModel.labels[pair.first]);
            result.add(String.valueOf(pair.second));
        }

        String[] resultString = result.toArray(new String[result.size()]);

        releaseTensors(inputTensors, outputTensors);

        return resultString;
    }

    private boolean prepareInputs(final Map<String, TF16UserBufferTensor> inputTensors,
                                  final Map<String, ByteBuffer> inputBuffers) {
        TensorAttributes inputAttributes = mNeuralNetwork.getTensorAttributes(mInputLayer);
        Tf16Params inputParams = resolveTf16Params(inputAttributes);

        inputBuffers.put(mInputLayer, ByteBuffer.allocateDirect(inputParams.size).order(ByteOrder.nativeOrder()));

        loadMeanImageIfAvailable(mModel.meanImage, inputParams.size);

        final int[] dimensions = inputAttributes.getDims();
        final boolean isGrayScale = (dimensions[dimensions.length -1] == 1);
        float[] imageBitmapAsFloat;
        if (!isGrayScale) {
            imageBitmapAsFloat = loadRgbBitmapAsFloat(mImage);
        } else {
            imageBitmapAsFloat = loadGrayScaleBitmapAsFloat(mImage);
        }
        quantize(imageBitmapAsFloat, inputBuffers.get(mInputLayer), inputParams);

        inputTensors.put(mInputLayer, mNeuralNetwork.createTF16UserBufferTensor(
                inputParams.size, inputParams.strides,
                inputParams.stepExactly0, inputParams.stepSize,
                inputBuffers.get(mInputLayer)));

        return true;
    }

    private void prepareOutputs(final Map<String, TF16UserBufferTensor> outputTensors,
                                final Map<String, ByteBuffer> outputBuffers) {
        TensorAttributes outputAttributes = mNeuralNetwork.getTensorAttributes(mOutputLayer);
        Tf16Params outputParams = resolveTf16Params(outputAttributes);
        outputParams.stepExactly0 = mStepExactly0;
        outputParams.stepSize = mStepSize;

        outputBuffers.put(mOutputLayer, ByteBuffer.allocateDirect(outputParams.size).order(ByteOrder.nativeOrder()));
        outputTensors.put(mOutputLayer, mNeuralNetwork.createTF16UserBufferTensor(
                outputParams.size, outputParams.strides,
                outputParams.stepExactly0, outputParams.stepSize,
                outputBuffers.get(mOutputLayer)));
    }

    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends UserBufferTensor>... tensorMaps) {
        for (Map<String, ? extends UserBufferTensor> tensorMap: tensorMaps) {
            for (UserBufferTensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    private void quantize(float[] src, ByteBuffer dst, Tf16Params tf16Params) {
        Tf16Encoding encoding = getTf16Encoding(src);


        short[] quantized = new short[src.length];
        //byte[] quantized = new byte[src.length];
        for (int i = 0; i < src.length; i++) {
            float data = Math.max(Math.min(src[i], encoding.max), encoding.min);
            data = data / encoding.delta - encoding.offset;
            quantized[i] = (short) Math.round(data);
        }

        for (short value : quantized) {
           dst.putShort(value);
        }
        //dst.putShort(quantized);
        //dst.put(quantized);
        tf16Params.stepSize = encoding.delta;
        tf16Params.stepExactly0 = Math.round(-encoding.min / encoding.delta);
    }

    private Tf16Encoding getTf16Encoding(float[] array) {
        Tf16Encoding encoding = new Tf16Encoding();

        int num_steps = (int) Math.pow(2, TF16_BITWIDTH) - 1;
        float new_min = Math.min(getMin(array), 0);
        float new_max = Math.max(getMax(array), 0);

        float min_range = 0.1f;
        new_max = Math.max(new_max, new_min + min_range);
        encoding.delta = (new_max - new_min) / num_steps;

        if (new_min < 0 && new_max > 0) {
            float quantized_zero = Math.round(-new_min / encoding.delta);
            quantized_zero = (float) Math.min(num_steps, Math.max(0.0, quantized_zero));
            encoding.offset = -quantized_zero;
        } else {
            encoding.offset = Math.round(new_min / encoding.delta);
        }

        encoding.min = encoding.delta * encoding.offset;
        encoding.max = encoding.delta * num_steps + encoding.min;

        return encoding;
    }

    private float[] dequantize(TF16UserBufferTensor tensor, ByteBuffer buffer) {
        final int outputSize = buffer.capacity();
        //final byte[] quantizedArray = new byte[outputSize];
        //buffer.getShort(quantizedArray);
        byte[] quantizedArray = new byte[outputSize];
        for (int i = 0; i < quantizedArray.length; i++) {
                    quantizedArray[i] = buffer.get();
        }
        //buffer.get(quantizedArray);

        final float[] dequantizedArray = new float[outputSize/2];
        int j=0;
        for (int i = 0; i < outputSize-1; i+=2) {
            String bin1 = String.format("%8s", Integer.toBinaryString(quantizedArray[i] & 0xFF)).replace(' ', '0');
            String bin2 = String.format("%8s", Integer.toBinaryString(quantizedArray[i + 1] & 0xFF)).replace(' ', '0');
            String concatenated = bin1 + bin2;
            int quantizedValue  = (int)Integer.parseInt(concatenated, 2) & 0xFFFF;
            //int quantizedValue = (int)quantizedArray[i] & 0xFFFF;
            dequantizedArray[j] = tensor.getMin() + quantizedValue *  tensor.getQuantizedStepSize();
            j = j+1 ;
        }
        return dequantizedArray;
    }

    private Tf16Params resolveTf16Params(TensorAttributes attribute) {
        int rank = attribute.getDims().length;
        int[] strides = new int[rank];
        strides[rank - 1] = TF16_SIZE;
        for (int i = rank - 1; i > 0; i--) {
            strides[i-1] = strides[i] * attribute.getDims()[i];
        }

        int bufferSize = TF16_SIZE;
        for (int dim: attribute.getDims()) {
            bufferSize *= dim;
        }

        return new Tf16Params(bufferSize, strides);
    }

    private class Tf16Params {
        int size;
        int[] strides;
        int stepExactly0;
        float stepSize;

        Tf16Params(int size, int[] strides) {
            this.size = size;
            this.strides = strides;
        }
    }

    private class Tf16Encoding {
        float min;
        float max;
        float delta;
        float offset;
    }

}