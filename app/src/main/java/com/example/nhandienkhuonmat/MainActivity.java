package com.example.nhandienkhuonmat;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;

import com.example.nhandienkhuonmat.ml.Model;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    PreviewView previewView;
    TextView tv_label;
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    Model model;
    List<String> labels;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initView();
        permission();
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            ProcessCameraProvider cameraProvider = null;
            try {
                cameraProvider = cameraProviderFuture.get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            bindPreview(cameraProvider);
        }, ContextCompat.getMainExecutor(this));
        instanceModel();
        realtimeRecogniton();


    }

    private void realtimeRecogniton(){
        Handler handler = new Handler();
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                float[] predict = predict();
                tv_label.setText(labels.get((int)predict[0])+": acc="+predict[1]);
                handler.postDelayed(this,500);
            }
        };
        handler.post(runnable);
    }
    //request permission
    public void permission(){
        ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},123);
    }
    //bind preview
    private void bindPreview(ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        Camera camera = cameraProvider.bindToLifecycle(this,cameraSelector,preview);

    }
    //instance model
    private void instanceModel(){
        try {
            model = Model.newInstance(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
        }
        getLabels();

    }
    private void getLabels(){
        labels = new ArrayList<>();
        BufferedReader  reader= null;
        try {
            reader = new BufferedReader(new InputStreamReader(MainActivity.this.getAssets().open("labels.txt")));
            String line;
            while ((line = reader.readLine())!=null){
                labels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //convert bitmap to bytebuffer
    final float IMAGE_MEAN = 127.5f;
    final float IMAGE_STD = 127.5f;
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap){
        //reize 224*224
        bitmap = Bitmap.createScaledBitmap(bitmap,224,224,false);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*224*224*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[]intValues = new int[224*224];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel  = 0;
        for(int i = 0;i<224;i++){
            for(int j  = 0;j<224; j++){
                int input = intValues[pixel++];

                byteBuffer.putFloat((((input >> 16 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input >> 8 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
            }
        }
        return byteBuffer;

    }
    //prediction
    private float[] predict(){
        float[]predict = new float[2];
        TensorBuffer inputFeatures = TensorBuffer.createFixedSize(new int[]{1,224,224,3}, DataType.FLOAT32);
        Bitmap bitmap =previewView.getBitmap();
        if(bitmap!=null){
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
            inputFeatures.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeatures);
            TensorBuffer outputBuffer = outputs.getOutputFeature0AsTensorBuffer();
            int index = findMax(outputBuffer.getFloatArray());
            predict[0] = index;
            predict[1] = outputBuffer.getFloatArray()[index];
        }
        return  predict;
    }
    private int findMax(float[]arr){
        int index = 0;
        float max = 0.0f;
        for(int i = 0;i<labels.size();i++){
            if(arr[i]>max) {
                max = arr[i];
                index = i;
            }
        }
        return index;
    }
    private void initView(){
        previewView= findViewById(R.id.preview);
        tv_label = findViewById(R.id.tv_labels);
    }
}