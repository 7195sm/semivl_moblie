package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.painter.BitmapPainter
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MyApplicationTheme
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyApplicationTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // Composable 함수 호출
                    SegmentationDisplay()
                }
            }
        }
    }

    // assets에서 이미지 파일을 로드하는 함수
    private fun loadImageFromAssets(context: Context, fileName: String): Bitmap {
        val assetManager = context.assets
        val inputStream: InputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }

    // 이미지를 전처리하는 함수
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 512, 512, true)
        val floatBuffer = ByteBuffer.allocateDirect(4 * 1 * 3 * 512 * 512).order(ByteOrder.nativeOrder()).asFloatBuffer()

        for (y in 0 until 512) {
            for (x in 0 until 512) {
                val pixel = resizedBitmap.getPixel(x, y)
                floatBuffer.put((pixel shr 16 and 0xFF) / 255.0f) // R
                floatBuffer.put((pixel shr 8 and 0xFF) / 255.0f)  // G
                floatBuffer.put((pixel and 0xFF) / 255.0f)       // B
            }
        }
        floatBuffer.rewind()
        return floatBuffer
    }

    // assets에서 모델 파일을 로컬 파일로 복사하는 함수
    private fun getModelFile(filename: String): File {
        val file = File(filesDir, filename)
        if (!file.exists()) {
            try {
                assets.open(filename).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        copyFile(inputStream, outputStream)
                    }
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        return file
    }

    @Throws(IOException::class)
    private fun copyFile(inputStream: InputStream, outputStream: FileOutputStream) {
        val buffer = ByteArray(1024)
        var read: Int
        while (inputStream.read(buffer).also { read = it } != -1) {
            outputStream.write(buffer, 0, read)
        }
    }

    // Segmentation 결과를 Bitmap으로 변환하는 함수
    private fun convertOutputToBitmap(output: Array<Array<Array<FloatArray>>>): Bitmap {
        val width = output[0][0][0].size
        val height = output[0][0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = output[0][0][y][x]
                val color = if (value > 0.5) 0xFFFF0000.toInt() else 0x00000000 // 예: 값이 0.5보다 크면 빨간색, 아니면 투명
                bitmap.setPixel(x, y, color)
            }
        }
        return bitmap
    }

    @Composable
    fun SegmentationDisplay() {
        val context = LocalContext.current
        var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var maskBitmap by remember { mutableStateOf<Bitmap?>(null) }

        LaunchedEffect(Unit) {
            try {
                val modelFile = getModelFile("semivl_model.onnx")
                val env = OrtEnvironment.getEnvironment()
                val session = env.createSession(modelFile.absolutePath)
                val inputName = session.inputNames.iterator().next()

                // 이미지를 불러와서 모델에 맞는 입력으로 변환
                val bitmap = loadImageFromAssets(context, "dog.jpg")
                originalBitmap = bitmap
                val inputData = preprocessImage(bitmap)

                // Create the input tensor
                val inputTensor = OnnxTensor.createTensor(env, inputData, longArrayOf(1, 3, 512, 512))

                // 모델 실행
                val results: OrtSession.Result = session.run(mapOf(inputName to inputTensor))
                val output = results[0].value

                // 출력 형식 처리
                if (output is Array<*>) {
                    val outputArray = output as Array<Array<Array<FloatArray>>>
                    maskBitmap = convertOutputToBitmap(outputArray)
                } else {
                    Log.e("ONNX", "Unexpected output type: ${output.javaClass.name}")
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error during ONNX model execution", e)
            }
        }

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            originalBitmap?.let {
                Image(
                    painter = BitmapPainter(it.asImageBitmap()),
                    contentDescription = null,
                    modifier = Modifier.size(256.dp)
                )
            }
            Spacer(modifier = Modifier.height(16.dp))
            maskBitmap?.let {
                Image(
                    painter = BitmapPainter(it.asImageBitmap()),
                    contentDescription = null,
                    modifier = Modifier.size(256.dp)
                )
            }
        }
    }

    @Composable
    fun Greeting(name: String) {
        Text(text = "Hello $name!")
    }

    @Preview(showBackground = true)
    @Composable
    fun DefaultPreview() {
        MyApplicationTheme {
            Greeting("Android")
        }
    }
}
