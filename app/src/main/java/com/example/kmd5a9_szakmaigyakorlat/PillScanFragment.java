package com.example.kmd5a9_szakmaigyakorlat;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.GradientDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;

import org.json.JSONObject;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.Executors;

public class PillScanFragment extends Fragment implements SensorEventListener {

    static {

        System.loadLibrary("opencv_java4");

    }

    private static final float LEVEL_TOLERANCE_DEGREES = 15.0f;

    private static final float MIN_FACE_UP_Z = 7.0f;

    private static final float SENSOR_SMOOTHING_ALPHA = 0.18f;

    private static final int CAMERA_TARGET_WIDTH_PX = 1600;
    private static final int CAMERA_TARGET_HEIGHT_PX = 1200;

    private static final double MAX_CAMERA_HEIGHT_CM = 25.0;

    private static final double HEIGHT_SMOOTHING_ALPHA = 0.30;

    private static final long HEIGHT_MEASUREMENT_MAX_AGE_MS = 1500L;

    private static final long LIVE_ARUCO_DIAGNOSTIC_DELAY_MS = 8000L;
    private static final int LIVE_ARUCO_DIAGNOSTIC_HISTORY_LIMIT = 20;

    private static final int SEGMENTED_BACKGROUND_GRAY = 145;
    private static final float SEGMENT_CROP_PADDING_RATIO = 0.10f;
    private static final int REFERENCE_ROUND_MEDICINE_LONGEST_SIDE_PX = 146;
    private static final int REFERENCE_ELONGATED_MEDICINE_LONGEST_SIDE_PX = 337;
    private static final double REFERENCE_ROUND_MEDICINE_ASPECT_RATIO = 1.0;
    private static final double REFERENCE_ELONGATED_MEDICINE_ASPECT_RATIO = 2.10;

    private static final int ROUND_TABLET_MIN_HULL_POINTS = 6;
    private static final double ROUND_TABLET_MAX_ASPECT_RATIO = 1.32;
    private static final double ROUND_TABLET_MIN_CIRCULARITY = 0.74;
    private static final double ROUND_TABLET_MIN_ELLIPSE_AREA_RATIO = 0.95;
    private static final double ROUND_TABLET_MAX_ELLIPSE_AREA_RATIO = 1.35;

    private static final double ROUND_TABLET_CIRCLE_MAX_BOX_ASPECT_RATIO = 1.30;
    private static final double ROUND_TABLET_CIRCLE_MIN_RADIUS_RATIO = 0.28;
    private static final double ROUND_TABLET_CIRCLE_MAX_RADIUS_RATIO = 0.65;
    private static final double ROUND_TABLET_CIRCLE_MAX_CENTER_OFFSET_RATIO = 0.23;
    private static final double ROUND_TABLET_CIRCLE_MIN_HULL_COVERAGE = 0.72;
    private static final double ROUND_TABLET_CIRCLE_MIN_CIRCLE_COVERAGE = 0.60;
    private static final double ROUND_TABLET_CIRCLE_MIN_EDGE_SUPPORT = 0.35;
    private static final double ROUND_TABLET_CIRCLE_RADIUS_SCALE = 0.98;
    private static final int ROUND_TABLET_CIRCLE_SAMPLE_COUNT = 96;

    private static final double CAPSULE_MIN_GEMINI_HULL_ASPECT_RATIO = 1.45;
    private static final double CAPSULE_MIN_IMAGE_ASPECT_RATIO = 1.45;
    private static final double CAPSULE_MAX_IMAGE_ASPECT_RATIO = 4.80;
    private static final double CAPSULE_MAX_CENTER_OFFSET_RATIO = 0.42;
    private static final double CAPSULE_MIN_CONTOUR_FILL_RATIO = 0.50;
    private static final double CAPSULE_MAX_CONTOUR_FILL_RATIO = 1.18;
    private static final double CAPSULE_MIN_GEMINI_HULL_COVERAGE = 0.62;
    private static final double CAPSULE_MIN_CANDIDATE_COVERAGE = 0.32;
    private static final double CAPSULE_LENGTH_SCALE = 0.98;
    private static final double CAPSULE_WIDTH_SCALE = 0.94;
    private static final double CAPSULE_ROI_PADDING_RATIO = 0.30;

    private static final double CAPSULE_COLOR_SEED_LENGTH_SCALE = 0.55;
    private static final double CAPSULE_COLOR_SEED_WIDTH_SCALE = 0.42;
    private static final double CAPSULE_MIN_LAB_FOREGROUND_BACKGROUND_DISTANCE = 18.0;
    private static final double CAPSULE_LAB_FOREGROUND_TO_BACKGROUND_RATIO = 0.95;
    private static final double CAPSULE_LAB_MAX_DISTANCE_SCALE = 0.70;
    private static final double CAPSULE_LAB_MIN_MAX_DISTANCE = 24.0;
    private static final double CAPSULE_LAB_MAX_MAX_DISTANCE = 75.0;

    private static final double OVAL_TABLET_MAX_BOX_ASPECT_RATIO = 1.70;
    private static final double OVAL_TABLET_MIN_FITTED_ASPECT_RATIO = 1.25;
    private static final double OVAL_TABLET_MAX_FITTED_ASPECT_RATIO = 1.85;
    private static final double OVAL_TABLET_MAX_CENTER_OFFSET_RATIO = 0.32;
    private static final double OVAL_TABLET_MIN_MAJOR_TO_BOX_RATIO = 0.55;
    private static final double OVAL_TABLET_MAX_MAJOR_TO_BOX_RATIO = 1.20;
    private static final double OVAL_TABLET_MIN_MINOR_TO_BOX_RATIO = 0.42;
    private static final double OVAL_TABLET_MAX_MINOR_TO_BOX_RATIO = 1.15;
    private static final double OVAL_TABLET_MIN_BOX_AREA_RATIO = 0.22;
    private static final double OVAL_TABLET_MAX_BOX_AREA_RATIO = 0.88;
    private static final double OVAL_TABLET_MIN_EDGE_SUPPORT = 0.45;
    private static final double OVAL_TABLET_MIN_HULL_COVERAGE = 0.30;
    private static final double OVAL_TABLET_MAJOR_SCALE = 0.98;
    private static final double OVAL_TABLET_MINOR_SCALE = 0.88;
    private static final double OVAL_TABLET_CENTER_BOX_WEIGHT = 0.80;
    private static final double OVAL_TABLET_ROI_PADDING_RATIO = 0.36;
    private static final int OVAL_TABLET_EDGE_SAMPLE_COUNT = 120;

    private static final double ELLIPSE_TABLET_MIN_ORIENTED_ASPECT_RATIO = 1.38;
    private static final double ELLIPSE_TABLET_MAX_ORIENTED_ASPECT_RATIO = 1.95;
    private static final double ELLIPSE_TABLET_MIN_FITTED_ASPECT_RATIO = 1.45;
    private static final double ELLIPSE_TABLET_MAX_FITTED_ASPECT_RATIO = 2.15;
    private static final double ELLIPSE_TABLET_MIN_CIRCULARITY = 0.70;
    private static final double ELLIPSE_TABLET_MAX_CIRCULARITY = 0.90;
    private static final double ELLIPSE_TABLET_MIN_BOUNDING_FILL_RATIO = 0.58;
    private static final double ELLIPSE_TABLET_MAX_BOUNDING_FILL_RATIO = 0.78;
    private static final double ELLIPSE_TABLET_MIN_ELLIPSE_AREA_RATIO = 0.94;
    private static final double ELLIPSE_TABLET_MAX_ELLIPSE_AREA_RATIO = 1.12;
    private static final double ELLIPSE_TABLET_MIN_HULL_COVERAGE = 0.70;
    private static final double ELLIPSE_TABLET_MIN_CANDIDATE_COVERAGE = 0.78;
    private static final double ELLIPSE_TABLET_MAJOR_SCALE = 0.98;
    private static final double ELLIPSE_TABLET_MINOR_SCALE = 0.88;
    private static final double ELLIPSE_TABLET_CENTER_SHIFT_WEIGHT = 0.70;
    private static final double ELLIPSE_TABLET_MAX_CENTER_SHIFT_MINOR_RATIO = 0.22;
    private static final double ELLIPSE_TABLET_ROI_PADDING_RATIO = 0.35;
    private static final double ELLIPSE_TABLET_BACKGROUND_BORDER_RATIO = 0.10;
    private static final double ELLIPSE_TABLET_MIN_AVERAGE_BRIGHTNESS_WEIGHT = 1.50;
    private static final double ELLIPSE_TABLET_MAX_PIXEL_BRIGHTNESS_WEIGHT = 16.0;

    private static final double TRIANGLE_TABLET_MAX_BOX_ASPECT_RATIO = 1.65;
    private static final double TRIANGLE_TABLET_ROI_PADDING_RATIO = 0.38;
    private static final double TRIANGLE_TABLET_SEED_RADIUS_RATIO = 0.16;
    private static final double TRIANGLE_TABLET_MIN_LAB_FOREGROUND_BACKGROUND_DISTANCE = 9.0;
    private static final double TRIANGLE_TABLET_LAB_LIGHTNESS_WEIGHT = 0.35;
    private static final double TRIANGLE_TABLET_LAB_FOREGROUND_TO_BACKGROUND_RATIO = 0.93;
    private static final double TRIANGLE_TABLET_CHROMA_FOREGROUND_TO_BACKGROUND_RATIO = 0.88;
    private static final double TRIANGLE_TABLET_LAB_MAX_DISTANCE_SCALE = 1.15;
    private static final double TRIANGLE_TABLET_LAB_MIN_MAX_DISTANCE = 16.0;
    private static final double TRIANGLE_TABLET_LAB_MAX_MAX_DISTANCE = 70.0;
    private static final double TRIANGLE_TABLET_MIN_CONTOUR_AREA_TO_BOX_RATIO = 0.12;
    private static final double TRIANGLE_TABLET_MAX_CONTOUR_AREA_TO_BOX_RATIO = 1.15;
    private static final double TRIANGLE_TABLET_MAX_ORIENTED_ASPECT_RATIO = 1.70;
    private static final double TRIANGLE_TABLET_MIN_SOLIDITY = 0.72;
    private static final double TRIANGLE_TABLET_MIN_HULL_FILL_RATIO = 0.38;
    private static final double TRIANGLE_TABLET_MAX_HULL_FILL_RATIO = 0.84;
    private static final double TRIANGLE_TABLET_APPROX_EPSILON_RATIO = 0.05;
    private static final double TRIANGLE_TABLET_MIN_TRIANGLE_TO_HULL_AREA_RATIO = 0.52;
    private static final double TRIANGLE_TABLET_MAX_TRIANGLE_TO_HULL_AREA_RATIO = 0.86;
    private static final double TRIANGLE_TABLET_MIN_VERTEX_ANGLE_DEGREES = 25.0;
    private static final double TRIANGLE_TABLET_MAX_VERTEX_ANGLE_DEGREES = 120.0;
    private static final double TRIANGLE_TABLET_MAX_SIDE_RATIO = 2.20;
    private static final double TRIANGLE_TABLET_MAX_CENTER_OFFSET_RATIO = 0.42;
    private static final double TRIANGLE_TABLET_MIN_GEMINI_HULL_COVERAGE = 0.20;
    private static final double TRIANGLE_TABLET_MIN_CANDIDATE_COVERAGE = 0.06;

    private static final double SEGMENT_SEARCH_WIDTH_MM = 100.0;
    private static final double SEGMENT_SEARCH_HEIGHT_MM = 60.0;

    private static final double SEGMENT_SEARCH_CENTER_OFFSET_Y_MM = 0.0;
    private static final double MINIMUM_MATCH_SCORE = 0.25;

    private static final int PARALLEL_RECOGNITION_THREADS = 3;

    private static final int TOP_K_MAX_CANDIDATES = 5;

    private static final String TOP_K_PREFERENCES =
            "top_k_accuracy_statistics";
    private SensorManager sensorManager;

    private Sensor accelerometer;

    private boolean isPhoneLevel = false;

    private float filteredX = 0f, filteredY = 0f, filteredZ = 9.81f;

    private boolean hasSensorFilter = false;

    private ImageView ivSelectedPill;

    private SeekBar sliderContrast;
    private FloatingActionButton btnFlash;

    private PreviewView viewFinder;

    private TextView tvResultName, tvCameraStatus;

    private Button btnOpenCamera, btnOpenGallery, btnTakePhotoFull, btnCloseCamera;

    private View levelIndicator, fullCameraContainer;

    private ImageCapture imageCapture;

    private boolean cameraStarted = false, isTakingPhoto = false;
    private volatile boolean isProcessing = false;

    private ActivityResultLauncher<String> galleryLauncher;

    private Bitmap currentBitmap;
    private Bitmap processedPillBitmap;

    private int contrastProgress = 50;
    private boolean isFlashSimulated = false;
    private boolean hasEditablePillPhoto = false;

    private String currentEvaluationFilename = "";

    private long lastArucoCheckTime = 0;

    private final Object liveArucoDiagnosticLock =
            new Object();

    private final List<LiveArucoFrameHistory>
            liveArucoDiagnosticHistory =
            new ArrayList<>();

    private long liveArucoDiagnosticStartedElapsed = 0L;
    private boolean liveArucoDiagnosticSaved = false;
    private boolean liveArucoDiagnosticSaving = false;
    private boolean liveArucoEverFoundAll = false;
    private String liveArucoDiagnosticSessionName = "";

    private boolean lastArucoState = false;
    private boolean isHeightMeasurementValid = false;
    private double filteredCameraHeightCm = Double.NaN;
    private double capturedCameraHeightCm = Double.NaN;
    private long lastHeightMeasurementElapsed = 0L;
    private String lastHeightMeasurementMessage = "";

    private volatile ArUcoHeightEstimator.CameraCalibration cameraCalibration;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private final ExecutorService recognitionExecutor =
            Executors.newFixedThreadPool(
                    PARALLEL_RECOGNITION_THREADS
            );

    private final Handler handler = new Handler(Looper.getMainLooper());

    private OnBackPressedCallback cameraBackCallback;

    @Override

    public void onCreate(@Nullable Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        if (!org.opencv.android.OpenCVLoader.initDebug()) Log.e("OpenCV", "Init hiba!");

        initLaunchers();

        sensorManager = (SensorManager) requireActivity().getSystemService(Context.SENSOR_SERVICE);

        if (sensorManager != null) accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

    }

    private void initLaunchers() {

        galleryLauncher = registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {

            if (uri != null) {

                try {

                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(requireActivity().getContentResolver(), uri);

                    currentEvaluationFilename =
                            getDisplayNameFromUri(uri);
                    capturedCameraHeightCm = Double.NaN;

                    preparePillPhotoForEditing(bitmap);

                } catch (Exception e) {
                    Log.e(
                            "PillScan",
                            "A galériából választott kép betöltése sikertelen.",
                            e
                    );

                    showError(
                            "A kiválasztott kép nem nyitható meg. "
                                    + "Válassz másik képet, vagy készíts "
                                    + "új fényképet.",
                            e
                    );
                }

            }

        });

    }

    @Override

    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_pill_scan, container, false);

        ivSelectedPill = view.findViewById(R.id.ivSelectedPill);
        sliderContrast = view.findViewById(R.id.slider_contrast);
        btnFlash = view.findViewById(R.id.btn_flash);

        if (ivSelectedPill != null) {
            ivSelectedPill.setScaleType(ImageView.ScaleType.FIT_CENTER);
            ivSelectedPill.setAdjustViewBounds(true);
        }

        viewFinder = view.findViewById(R.id.viewFinder);

        tvResultName = view.findViewById(R.id.tvResultName);

        tvCameraStatus = view.findViewById(R.id.tvCameraStatus);

        btnOpenCamera = view.findViewById(R.id.btnOpenCamera);

        btnOpenGallery = view.findViewById(R.id.btnOpenGallery);

        btnTakePhotoFull = view.findViewById(R.id.btnTakePhotoFull);

        btnCloseCamera = view.findViewById(R.id.btnCloseCamera);

        levelIndicator = view.findViewById(R.id.levelIndicator);

        fullCameraContainer = view.findViewById(R.id.fullCameraContainer);

        setupLevelIndicator();

        btnOpenCamera.setOnClickListener(v -> {
            if (isProcessing) {
                showError("A gyógyszerfelismerés már folyamatban van.");
                return;
            }

            if (hasEditablePillPhoto) {
                startRecognitionFromEditedPhoto();
            } else {
                openFullScreenCamera();
            }
        });

        btnOpenGallery.setOnClickListener(v -> {
            if (cameraStarted) {
                closeFullScreenCamera();
            }

            galleryLauncher.launch("image/*");
        });

        setupPillImageEditor();

        if (btnTakePhotoFull != null) btnTakePhotoFull.setOnClickListener(v -> takePhotoIfReady());

        if (btnCloseCamera != null) {
            btnCloseCamera.setText("← Vissza");
            btnCloseCamera.setOnClickListener(
                    v -> closeFullScreenCamera()
            );
        }

        return view;

    }

    private void setupPillImageEditor() {
        if (sliderContrast != null) {
            sliderContrast.setMax(100);
            sliderContrast.setProgress(50);
            sliderContrast.setEnabled(false);

            sliderContrast.setOnSeekBarChangeListener(
                    new SeekBar.OnSeekBarChangeListener() {
                        @Override
                        public void onProgressChanged(
                                SeekBar seekBar,
                                int progress,
                                boolean fromUser
                        ) {
                            contrastProgress = progress;

                            if (hasEditablePillPhoto
                                    && currentBitmap != null) {
                                applyPillLiveFilters();
                            }
                        }

                        @Override
                        public void onStartTrackingTouch(
                                SeekBar seekBar
                        ) {
                        }

                        @Override
                        public void onStopTrackingTouch(
                                SeekBar seekBar
                        ) {
                        }
                    }
            );
        }

        if (btnFlash != null) {
            btnFlash.setEnabled(false);
            updatePillFlashButtonColor();

            btnFlash.setOnClickListener(v -> {
                if (!hasEditablePillPhoto
                        || currentBitmap == null
                        || isProcessing) {
                    return;
                }

                isFlashSimulated = !isFlashSimulated;
                updatePillFlashButtonColor();
                applyPillLiveFilters();
            });
        }
    }

    private void preparePillPhotoForEditing(
            @NonNull Bitmap bitmap
    ) {
        if (bitmap.isRecycled()) {
            showError(
                    "A kiválasztott kép már nem érhető el. "
                            + "Válassz másik képet, vagy készíts "
                            + "új fényképet."
            );
            return;
        }

        if (processedPillBitmap != null
                && processedPillBitmap != bitmap
                && !processedPillBitmap.isRecycled()) {
            processedPillBitmap.recycle();
        }

        if (currentBitmap != null
                && currentBitmap != bitmap
                && !currentBitmap.isRecycled()) {
            currentBitmap.recycle();
        }

        currentBitmap = bitmap;
        processedPillBitmap = null;

        contrastProgress = 50;
        isFlashSimulated = false;
        hasEditablePillPhoto = true;

        if (sliderContrast != null) {
            sliderContrast.setProgress(50);
            sliderContrast.setEnabled(true);
        }

        if (btnFlash != null) {
            btnFlash.setEnabled(true);
        }

        updatePillFlashButtonColor();
        applyPillLiveFilters();

        if (ivSelectedPill != null) {
            ivSelectedPill.setVisibility(View.VISIBLE);
            ivSelectedPill.setScaleType(
                    ImageView.ScaleType.FIT_CENTER
            );
        }

        if (btnOpenCamera != null) {
            btnOpenCamera.setEnabled(true);
            btnOpenCamera.setText("Gyógyszer felismerése");
        }

        if (tvResultName != null) {
            tvResultName.setText(
                    "A kép elkészült. Állítsd be a kontrasztot "
                            + "és a fényerőt, majd nyomd meg a "
                            + "„Gyógyszer felismerése” gombot."
            );
        }
    }

    private void applyPillLiveFilters() {
        if (currentBitmap == null
                || currentBitmap.isRecycled()) {
            return;
        }

        try {
            float contrast =
                    contrastProgress / 50.0f;

            float brightness =
                    isFlashSimulated
                            ? 40.0f
                            : 0.0f;

            Bitmap filteredBitmap =
                    Bitmap.createBitmap(
                            currentBitmap.getWidth(),
                            currentBitmap.getHeight(),
                            Bitmap.Config.ARGB_8888
                    );

            Canvas canvas =
                    new Canvas(filteredBitmap);

            Paint paint =
                    new Paint(
                            Paint.ANTI_ALIAS_FLAG
                                    | Paint.FILTER_BITMAP_FLAG
                    );

            float translation =
                    brightness
                            + 128.0f
                            * (1.0f - contrast);

            ColorMatrix colorMatrix =
                    new ColorMatrix(
                            new float[]{
                                    contrast, 0, 0, 0, translation,
                                    0, contrast, 0, 0, translation,
                                    0, 0, contrast, 0, translation,
                                    0, 0, 0, 1, 0
                            }
                    );

            paint.setColorFilter(
                    new ColorMatrixColorFilter(
                            colorMatrix
                    )
            );

            canvas.drawBitmap(
                    currentBitmap,
                    0,
                    0,
                    paint
            );

            Bitmap previousProcessed =
                    processedPillBitmap;

            processedPillBitmap =
                    filteredBitmap;

            if (ivSelectedPill != null) {
                ivSelectedPill.setLayerType(
                        View.LAYER_TYPE_SOFTWARE,
                        null
                );
                ivSelectedPill.setBackgroundColor(
                        Color.TRANSPARENT
                );
                ivSelectedPill.setScaleType(
                        ImageView.ScaleType.FIT_CENTER
                );
                ivSelectedPill.setImageBitmap(
                        processedPillBitmap
                );
            }

            if (previousProcessed != null
                    && previousProcessed
                    != processedPillBitmap
                    && !previousProcessed.isRecycled()) {
                previousProcessed.recycle();
            }

        } catch (OutOfMemoryError memoryError) {
            Log.e(
                    "PillScan",
                    "Nincs elég memória a gyógyszerkép "
                            + "szűréséhez.",
                    memoryError
            );

            if (ivSelectedPill != null) {
                ivSelectedPill.setImageBitmap(
                        currentBitmap
                );
            }

            showError(
                    "A kép túl nagy a készülék számára. "
                            + "Válassz kisebb felbontású képet, "
                            + "vagy készíts új fényképet."
            );

        } catch (Exception error) {
            Log.e(
                    "PillScan",
                    "A gyógyszerkép szűrése sikertelen.",
                    error
            );

            showError(
                    "A kép megjelenítése nem sikerült. "
                            + "Válassz kisebb felbontású képet, "
                            + "vagy készíts új fényképet.",
                    error
            );
        }
    }

    private void updatePillFlashButtonColor() {
        if (btnFlash == null) {
            return;
        }

        int color =
                isFlashSimulated
                        ? Color.YELLOW
                        : Color.DKGRAY;

        btnFlash.setBackgroundTintList(
                ColorStateList.valueOf(color)
        );
    }

    private void startRecognitionFromEditedPhoto() {
        if (currentBitmap == null
                || currentBitmap.isRecycled()) {
            showError(
                    "A fénykép már nem érhető el. "
                            + "Készíts új képet, vagy válassz "
                            + "másikat a galériából."
            );
            return;
        }

        if (processedPillBitmap == null
                || processedPillBitmap.isRecycled()) {
            showError(
                    "A beállított fényképet nem sikerült "
                            + "előkészíteni. Készíts új képet, "
                            + "vagy válassz másikat a galériából."
            );
            return;
        }

        Bitmap originalCopy =
                currentBitmap.copy(
                        Bitmap.Config.ARGB_8888,
                        false
                );

        Bitmap filteredCopy =
                processedPillBitmap.copy(
                        Bitmap.Config.ARGB_8888,
                        false
                );

        if (originalCopy == null
                || filteredCopy == null) {
            if (originalCopy != null
                    && !originalCopy.isRecycled()) {
                originalCopy.recycle();
            }

            if (filteredCopy != null
                    && !filteredCopy.isRecycled()) {
                filteredCopy.recycle();
            }

            showError(
                    "A fényképet nem sikerült előkészíteni "
                            + "a felismeréshez. Próbáld újra, "
                            + "vagy válassz másik képet."
            );
            return;
        }

        hasEditablePillPhoto = false;

        if (sliderContrast != null) {
            sliderContrast.setEnabled(false);
        }

        if (btnFlash != null) {
            btnFlash.setEnabled(false);
        }

        if (btnOpenCamera != null) {
            btnOpenCamera.setText(
                    "Feldolgozás folyamatban..."
            );
        }

        startProcessing(
                filteredCopy,
                originalCopy
        );
    }

    @Override
    public void onViewCreated(
            @NonNull View view,
            @Nullable Bundle savedInstanceState
    ) {
        super.onViewCreated(view, savedInstanceState);

        cameraBackCallback =
                new OnBackPressedCallback(true) {
                    @Override
                    public void handleOnBackPressed() {

                        if (isFullScreenCameraVisible()) {
                            if (isTakingPhoto) {
                                if (tvCameraStatus != null) {
                                    tvCameraStatus.setText(
                                            "A fénykép készítése folyamatban van. "
                                                    + "Kérlek, várj."
                                    );
                                }
                                return;
                            }

                            closeFullScreenCamera();
                            return;
                        }

                        setEnabled(false);
                        requireActivity()
                                .getOnBackPressedDispatcher()
                                .onBackPressed();

                        setEnabled(true);
                    }
                };

        requireActivity()
                .getOnBackPressedDispatcher()
                .addCallback(
                        getViewLifecycleOwner(),
                        cameraBackCallback
                );
    }

    private void checkCameraPermissionAndStart() {

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)

                == PackageManager.PERMISSION_GRANTED) {

            startCamera();

        } else {

            requestPermissionLauncher.launch(Manifest.permission.CAMERA);

        }

    }

    private final ActivityResultLauncher<String> requestPermissionLauncher =

            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {

                if (isGranted) {

                    startCamera();

                } else {

                    showError(
                            "A fényképezéshez engedélyezned kell a "
                                    + "kamera használatát az alkalmazás "
                                    + "beállításaiban."
                    );

                }

            });

    private Uri saveBitmapToGallery(Bitmap bitmap) {
        if (bitmap == null
                || bitmap.isRecycled()
                || bitmap.getWidth() <= 0
                || bitmap.getHeight() <= 0) {
            showError(
                    "A fényképet nem sikerült elmenteni, mert a "
                            + "kép nem olvasható. Készíts új fényképet."
            );
            return null;
        }

        String timestamp =
                new SimpleDateFormat(
                        "yyyyMMdd_HHmmss_SSS",
                        Locale.ROOT
                ).format(new Date());

        String filename =
                "Pill_" + timestamp + ".jpg";

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return savePillBitmapWithMediaStore(
                    bitmap,
                    filename
            );
        }

        return savePillBitmapLegacy(
                bitmap,
                filename
        );
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private Uri savePillBitmapWithMediaStore(
            Bitmap bitmap,
            String filename
    ) {
        Uri uri = null;

        try {
            ContentValues values =
                    new ContentValues();

            values.put(
                    MediaStore.MediaColumns.DISPLAY_NAME,
                    filename
            );

            values.put(
                    MediaStore.MediaColumns.MIME_TYPE,
                    "image/jpeg"
            );

            values.put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    Environment.DIRECTORY_DOWNLOADS
                            + "/MedScan/Images/Pills"
            );

            values.put(
                    MediaStore.MediaColumns.IS_PENDING,
                    1
            );

            uri = requireContext()
                    .getContentResolver()
                    .insert(
                            MediaStore.Downloads
                                    .EXTERNAL_CONTENT_URI,
                            values
                    );

            if (uri == null) {
                throw new IOException(
                        "A MediaStore nem hozott létre fájlt."
                );
            }

            try (OutputStream outputStream =
                         requireContext()
                                 .getContentResolver()
                                 .openOutputStream(uri)) {
                if (outputStream == null) {
                    throw new IOException(
                            "A képfájl nem nyitható meg."
                    );
                }

                boolean saved =
                        bitmap.compress(
                                Bitmap.CompressFormat.JPEG,
                                100,
                                outputStream
                        );

                if (!saved) {
                    throw new IOException(
                            "A JPEG-kép létrehozása sikertelen."
                    );
                }

                outputStream.flush();
            }

            ContentValues finished =
                    new ContentValues();

            finished.put(
                    MediaStore.MediaColumns.IS_PENDING,
                    0
            );

            requireContext()
                    .getContentResolver()
                    .update(
                            uri,
                            finished,
                            null,
                            null
                    );

            Log.d(
                    "PillScan",
                    "Kép sikeresen mentve: "
                            + "Download/MedScan/Images/Pills/"
                            + filename
            );

            showSavedFilename(filename);
            return uri;

        } catch (Exception saveError) {
            Log.e(
                    "PillScan",
                    "Gyógyszerfotó MediaStore-mentési hiba",
                    saveError
            );

            if (uri != null) {
                try {
                    requireContext()
                            .getContentResolver()
                            .delete(
                                    uri,
                                    null,
                                    null
                            );
                } catch (Exception deleteError) {
                    Log.w(
                            "PillScan",
                            "A hibás MediaStore-bejegyzés "
                                    + "nem törölhető.",
                            deleteError
                    );
                }
            }

            handler.post(() ->
                    showError(
                            "A gyógyszerfotó mentése nem sikerült, "
                                    + "de a felismerés folytatódik."
                    )
            );

            return null;
        }
    }

    private Uri savePillBitmapLegacy(
            Bitmap bitmap,
            String filename
    ) {
        try {
            File downloadsRoot =
                    requireContext()
                            .getExternalFilesDir(
                                    Environment.DIRECTORY_DOWNLOADS
                            );

            if (downloadsRoot == null) {
                throw new IOException(
                        "A külső Download könyvtár nem érhető el."
                );
            }

            File pillsDirectory =
                    new File(
                            downloadsRoot,
                            "MedScan/Images/Pills"
                    );

            if (!pillsDirectory.exists()
                    && !pillsDirectory.mkdirs()) {
                throw new IOException(
                        "A gyógyszermappa nem hozható létre."
                );
            }

            File target =
                    new File(
                            pillsDirectory,
                            filename
                    );

            try (OutputStream outputStream =
                         new FileOutputStream(target)) {
                boolean saved =
                        bitmap.compress(
                                Bitmap.CompressFormat.JPEG,
                                100,
                                outputStream
                        );

                if (!saved) {
                    throw new IOException(
                            "A JPEG-kép létrehozása sikertelen."
                    );
                }

                outputStream.flush();
            }

            Log.d(
                    "PillScan",
                    "Régi Androidon kép mentve: "
                            + target.getAbsolutePath()
            );

            showSavedFilename(filename);
            return Uri.fromFile(target);

        } catch (Exception saveError) {
            Log.e(
                    "PillScan",
                    "Gyógyszerfotó legacy mentési hiba",
                    saveError
            );

            handler.post(() ->
                    showError(
                            "A gyógyszerfotó mentése nem sikerült, "
                                    + "de a felismerés folytatódik."
                    )
            );

            return null;
        }
    }

    private void openFullScreenCamera() {

        if (fullCameraContainer == null) {

            showError(
                    "A kamera felülete nem tölthető be. "
                            + "Indítsd újra az alkalmazást.",
                    new IllegalStateException(
                            "Hiányzik az XML-ből a fullCameraContainer."
                    )
            );

            return;

        }

        if (viewFinder == null) {

            showError(
                    "A kamera előnézete nem tölthető be. "
                            + "Indítsd újra az alkalmazást.",
                    new IllegalStateException(
                            "Hiányzik az XML-ből a viewFinder."
                    )
            );

            return;

        }

        if (accelerometer == null) {

            showError(
                    "Ezen a készüléken nem ellenőrizhető a telefon "
                            + "vízszintes helyzete, ezért a fényképezés "
                            + "nem indítható el."
            );

            return;

        }

        resetHeightMeasurementState();
        capturedCameraHeightCm = Double.NaN;

        fullCameraContainer.setVisibility(View.VISIBLE);

        if (ivSelectedPill != null) {

            ivSelectedPill.setVisibility(View.GONE);

        }

        if (tvCameraStatus != null) {

            tvCameraStatus.setText(
                    BuildConfig.DEBUG
                            ? "Igazítsd vízszintbe a telefont, és "
                              + "mutasd a 0, 1, 2 és 3 ArUco-jelölést."
                            : "Igazítsd vízszintbe a telefont, és "
                              + "mutasd a lap mind a négy "
                              + "sarokjelölését."
            );

        }

        if (btnTakePhotoFull != null) {

            btnTakePhotoFull.setEnabled(false);

        }

        tvResultName.setText(
                BuildConfig.DEBUG
                        ? "Kamera megnyitva. Zöld jelzésnél készíthető "
                          + "fotó. Ha a sarkok nem zöldülnek ki, tartsd "
                          + "a teljes lapot 8 másodpercig a képen; az "
                          + "ArUco-diagnosztika automatikusan elmentődik."
                        : "Kamera megnyitva. Mutasd a teljes lapot és "
                          + "mind a négy sarokjelölést. Zöld jelzésnél "
                          + "elkészítheted a fényképet."
        );

        checkCameraPermissionAndStart();

    }

    private void closeFullScreenCamera() {

        if (fullCameraContainer != null) {

            fullCameraContainer.setVisibility(View.GONE);

        }

        stopCamera();

        cameraStarted = false;

        imageCapture = null;

        isTakingPhoto = false;
        resetHeightMeasurementState();
        cameraCalibration = null;

        if (btnTakePhotoFull != null) {

            btnTakePhotoFull.setEnabled(false);

        }

        if (btnCloseCamera != null) {

            btnCloseCamera.setEnabled(true);

        }

        if (btnOpenCamera != null) {

            btnOpenCamera.setEnabled(true);

            btnOpenCamera.setText("Kamera");

        }

        tvResultName.setText("Kamera bezárva.");

    }

    private void setupLevelIndicator() {

        updateLevelIndicator(false);

    }

    private void updateLevelIndicator(boolean level) {

        if (levelIndicator == null) return;

        GradientDrawable drawable = new GradientDrawable();

        drawable.setShape(GradientDrawable.OVAL);

        drawable.setColor(level ? Color.GREEN : Color.RED);

        drawable.setStroke(dpToPx(3), Color.WHITE);

        levelIndicator.setBackground(drawable);

        levelIndicator.setAlpha(0.95f);

    }

    private void startCamera() {

        ProcessCameraProvider.getInstance(requireContext()).addListener(() -> {

            try {

                ProcessCameraProvider provider =
                        ProcessCameraProvider.getInstance(requireContext()).get();

                viewFinder.setScaleType(
                        PreviewView.ScaleType.FIT_CENTER
                );

                if (ivSelectedPill != null) {
                    ivSelectedPill.setScaleType(
                            ImageView.ScaleType.FIT_CENTER
                    );
                }

                viewFinder.post(() -> {

                    if (!isAdded()
                            || getView() == null
                            || viewFinder == null) {
                        return;
                    }

                    try {

                        ResolutionSelector fourThreeResolutionSelector =
                                new ResolutionSelector.Builder()
                                        .setAspectRatioStrategy(
                                                new AspectRatioStrategy(
                                                        AspectRatio.RATIO_4_3,
                                                        AspectRatioStrategy
                                                                .FALLBACK_RULE_AUTO
                                                )
                                        )
                                        .setResolutionStrategy(
                                                new ResolutionStrategy(
                                                        new Size(
                                                                CAMERA_TARGET_WIDTH_PX,
                                                                CAMERA_TARGET_HEIGHT_PX
                                                        ),
                                                        ResolutionStrategy
                                                                .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                                )
                                        )
                                        .build();

                        int targetRotation =
                                viewFinder.getDisplay() == null
                                        ? Surface.ROTATION_0
                                        : viewFinder.getDisplay()
                                          .getRotation();

                        Preview preview =
                                new Preview.Builder()
                                        .setResolutionSelector(
                                                fourThreeResolutionSelector
                                        )
                                        .setTargetRotation(
                                                targetRotation
                                        )
                                        .build();

                        preview.setSurfaceProvider(
                                viewFinder.getSurfaceProvider()
                        );

                        ImageAnalysis imageAnalysis =
                                new ImageAnalysis.Builder()
                                        .setResolutionSelector(
                                                fourThreeResolutionSelector
                                        )
                                        .setTargetRotation(
                                                targetRotation
                                        )
                                        .setBackpressureStrategy(
                                                ImageAnalysis
                                                        .STRATEGY_KEEP_ONLY_LATEST
                                        )
                                        .build();

                        imageCapture =
                                new ImageCapture.Builder()
                                        .setResolutionSelector(
                                                fourThreeResolutionSelector
                                        )
                                        .setTargetRotation(
                                                targetRotation
                                        )
                                        .setCaptureMode(
                                                ImageCapture
                                                        .CAPTURE_MODE_MINIMIZE_LATENCY
                                        )
                                        .build();

                        CameraSelector cameraSelector =
                                CameraSelector.DEFAULT_BACK_CAMERA;

                        provider.unbindAll();

                        ViewPort sharedViewPort =
                                viewFinder.getViewPort(
                                        targetRotation
                                );

                        UseCaseGroup.Builder useCaseGroupBuilder =
                                new UseCaseGroup.Builder()
                                        .addUseCase(preview)
                                        .addUseCase(imageCapture)
                                        .addUseCase(imageAnalysis);

                        if (sharedViewPort != null) {
                            useCaseGroupBuilder.setViewPort(
                                    sharedViewPort
                            );
                        } else {
                            Log.w(
                                    "PillScan",
                                    "A PreviewView ViewPort még nem érhető el. "
                                            + "A 4:3-as use case beállítás "
                                            + "ettől még érvényben marad."
                            );
                        }

                        Camera boundCamera =
                                provider.bindToLifecycle(
                                        getViewLifecycleOwner(),
                                        cameraSelector,
                                        useCaseGroupBuilder.build()
                                );

                        try {
                            boundCamera.getCameraControl()
                                    .setZoomRatio(1.0f);
                        } catch (Exception zoomError) {
                            Log.w(
                                    "PillScan",
                                    "Az 1x zoom nem állítható be.",
                                    zoomError
                            );
                        }

                        try {
                            cameraCalibration =
                                    ArUcoHeightEstimator
                                            .createCalibration(
                                                    boundCamera
                                                            .getCameraInfo()
                                            );
                        } catch (Exception calibrationError) {
                            cameraCalibration = null;

                            lastHeightMeasurementMessage =
                                    calibrationError.getMessage() == null
                                            ? "A kamera nem kalibrálható."
                                            : calibrationError.getMessage();

                            Log.e(
                                    "PillScan",
                                    "Kamerakalibrációs hiba",
                                    calibrationError
                            );
                        }

                        resetLiveArucoDiagnosticSession();

                        imageAnalysis.setAnalyzer(
                                executor,
                                image -> {
                                    long currentTime =
                                            SystemClock.elapsedRealtime();

                                    if (currentTime
                                            - lastArucoCheckTime
                                            < 500L) {
                                        image.close();
                                        return;
                                    }

                                    lastArucoCheckTime =
                                            currentTime;

                                    Mat mat = null;

                                    boolean allMarkersFound =
                                            false;

                                    LiveArucoDetectionResult
                                            detectionResult =
                                            null;

                                    ArUcoHeightEstimator.HeightMeasurement
                                            measurement;

                                    try {

                                        mat =
                                                imageProxyToGrayscaleMatForLiveAruco(
                                                        image
                                                );

                                        ImageProxy.PlaneProxy yPlane =
                                                image.getPlanes()[0];

                                        Log.d(
                                                "LiveAruco",
                                                "ImageProxy="
                                                        + image.getWidth()
                                                        + "x"
                                                        + image.getHeight()
                                                        + "; rotation="
                                                        + image.getImageInfo()
                                                        .getRotationDegrees()
                                                        + "; cropRect="
                                                        + image.getCropRect()
                                                        + "; Y rowStride="
                                                        + yPlane.getRowStride()
                                                        + "; Y pixelStride="
                                                        + yPlane.getPixelStride()
                                                        + "; analyzerMat="
                                                        + mat.cols()
                                                        + "x"
                                                        + mat.rows()
                                                        + "; channels="
                                                        + mat.channels()
                                        );

                                        detectionResult =
                                                detectLiveArucoMarkers(
                                                        mat
                                                );

                                        allMarkersFound =
                                                detectionResult
                                                        .allRequiredFound;

                                        recordLiveArucoDiagnosticFrame(
                                                currentTime,
                                                image,
                                                mat,
                                                detectionResult
                                        );

                                        maybeSaveLiveArucoDiagnostic(
                                                currentTime,
                                                image,
                                                mat,
                                                detectionResult
                                        );

                                        if (allMarkersFound) {
                                            measurement =
                                                    ArUcoHeightEstimator
                                                            .estimateHeight(
                                                                    mat,
                                                                    cameraCalibration
                                                            );
                                        } else {
                                            measurement =
                                                    ArUcoHeightEstimator
                                                            .HeightMeasurement
                                                            .invalid(
                                                                    false,
                                                                    "Nem látható mind a négy "
                                                                            + "szükséges ArUco-kód."
                                                            );
                                        }

                                    } catch (Exception analyzerError) {
                                        measurement =
                                                ArUcoHeightEstimator
                                                        .HeightMeasurement
                                                        .invalid(
                                                                allMarkersFound,
                                                                "Kamerakép-elemzési "
                                                                        + "hiba: "
                                                                        + analyzerError
                                                                        .getMessage()
                                                        );

                                    } finally {
                                        if (mat != null) {
                                            mat.release();
                                        }

                                        image.close();
                                    }

                                    final boolean
                                            finalAllMarkersFound =
                                            allMarkersFound;

                                    ArUcoHeightEstimator.HeightMeasurement
                                            finalMeasurement =
                                            measurement;

                                    handler.post(() ->
                                            updateCameraMeasurement(
                                                    finalMeasurement,
                                                    finalAllMarkersFound
                                            )
                                    );
                                }
                        );

                        cameraStarted = true;
                        refreshCameraReadyUi();

                    } catch (Exception bindError) {
                        Log.e(
                                "PillScan",
                                "CameraX 4:3 kötési hiba",
                                bindError
                        );

                        showError(
                                "A kamera nem indítható el. Zárd be a "
                                        + "kamerát használó más "
                                        + "alkalmazásokat, majd próbáld újra.",
                                bindError
                        );
                    }
                });

            } catch (Exception providerError) {
                Log.e(
                        "PillScan",
                        "CameraX provider hiba",
                        providerError
                );

                showError(
                        "A kamera jelenleg nem érhető el. Zárd be a "
                                + "kamerát használó más alkalmazásokat, "
                                + "majd próbáld újra.",
                        providerError
                );
            }

        }, ContextCompat.getMainExecutor(requireContext()));

    }

    private Mat imageProxyToGrayscaleMatForLiveAruco(
            @NonNull ImageProxy image
    ) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException(
                    "Az élő ArUco-elemzés YUV_420_888 képet vár, kapott formátum: "
                            + image.getFormat()
            );
        }

        ImageProxy.PlaneProxy[] planes =
                image.getPlanes();

        if (planes == null || planes.length == 0) {
            throw new IllegalStateException(
                    "Az ImageProxy nem tartalmaz Y-síkot."
            );
        }

        ImageProxy.PlaneProxy yPlane =
                planes[0];

        ByteBuffer yBuffer =
                yPlane.getBuffer()
                        .duplicate();

        yBuffer.rewind();

        int width =
                image.getWidth();

        int height =
                image.getHeight();

        int rowStride =
                yPlane.getRowStride();

        int pixelStride =
                yPlane.getPixelStride();

        if (width <= 0
                || height <= 0
                || rowStride <= 0
                || pixelStride <= 0) {
            throw new IllegalStateException(
                    "Hibás Y-sík méret vagy stride. "
                            + "width="
                            + width
                            + ", height="
                            + height
                            + ", rowStride="
                            + rowStride
                            + ", pixelStride="
                            + pixelStride
            );
        }

        byte[] grayPixels =
                new byte[
                        width * height
                        ];

        int bufferLimit =
                yBuffer.limit();

        for (int row = 0;
             row < height;
             row++) {
            int sourceRowStart =
                    row * rowStride;

            int destinationRowStart =
                    row * width;

            if (pixelStride == 1) {
                int lastRequiredIndex =
                        sourceRowStart
                                + width
                                - 1;

                if (lastRequiredIndex >= bufferLimit) {
                    throw new IllegalStateException(
                            "Az Y-sík rövidebb a vártnál a(z) "
                                    + row
                                    + ". sorban. limit="
                                    + bufferLimit
                                    + ", szükséges index="
                                    + lastRequiredIndex
                    );
                }

                ByteBuffer rowBuffer =
                        yBuffer.duplicate();

                rowBuffer.position(
                        sourceRowStart
                );

                rowBuffer.get(
                        grayPixels,
                        destinationRowStart,
                        width
                );

            } else {
                for (int column = 0;
                     column < width;
                     column++) {
                    int sourceIndex =
                            sourceRowStart
                                    + column
                                    * pixelStride;

                    if (sourceIndex >= bufferLimit) {
                        throw new IllegalStateException(
                                "Az Y-sík rövidebb a vártnál. "
                                        + "row="
                                        + row
                                        + ", column="
                                        + column
                                        + ", limit="
                                        + bufferLimit
                                        + ", szükséges index="
                                        + sourceIndex
                        );
                    }

                    grayPixels[
                            destinationRowStart
                                    + column
                            ] =
                            yBuffer.get(
                                    sourceIndex
                            );
                }
            }
        }

        Mat grayMat =
                new Mat(
                        height,
                        width,
                        CvType.CV_8UC1
                );

        grayMat.put(
                0,
                0,
                grayPixels
        );

        return grayMat;
    }

    private LiveArucoDetectionResult detectLiveArucoMarkers(
            Mat cameraFrame
    ) {
        if (cameraFrame == null
                || cameraFrame.empty()) {
            return LiveArucoDetectionResult.empty();
        }

        Dictionary dictionary =
                Objdetect.getPredefinedDictionary(
                        Objdetect.DICT_4X4_50
                );

        DetectorParameters parameters =
                new DetectorParameters();

        ArucoDetector detector =
                new ArucoDetector(
                        dictionary,
                        parameters
                );

        List<Mat> detectedCornerMats =
                new ArrayList<>();

        List<Mat> rejectedCornerMats =
                new ArrayList<>();

        Mat ids =
                new Mat();

        try {

            detector.detectMarkers(
                    cameraFrame,
                    detectedCornerMats,
                    ids,
                    rejectedCornerMats
            );

            boolean[] foundRequired =
                    new boolean[4];

            List<Integer> detectedIds =
                    new ArrayList<>();

            List<LiveArucoMarker> markers =
                    new ArrayList<>();

            int detectedCount =
                    Math.min(
                            ids.rows(),
                            detectedCornerMats.size()
                    );

            for (int row = 0;
                 row < detectedCount;
                 row++) {
                double[] idValue =
                        ids.get(
                                row,
                                0
                        );

                if (idValue == null
                        || idValue.length == 0) {
                    continue;
                }

                int markerId =
                        (int) Math.round(
                                idValue[0]
                        );

                detectedIds.add(
                        markerId
                );

                if (markerId >= 0
                        && markerId <= 3) {
                    foundRequired[markerId] =
                            true;
                }

                Point[] corners =
                        readLiveArucoCorners(
                                detectedCornerMats.get(row)
                        );

                if (corners != null) {
                    markers.add(
                            new LiveArucoMarker(
                                    markerId,
                                    corners
                            )
                    );
                }
            }

            List<Point[]> rejectedCandidates =
                    new ArrayList<>();

            for (Mat rejectedCornerMat
                    : rejectedCornerMats) {
                Point[] corners =
                        readLiveArucoCorners(
                                rejectedCornerMat
                        );

                if (corners != null) {
                    rejectedCandidates.add(
                            corners
                    );
                }
            }

            List<Integer> missingIds =
                    new ArrayList<>();

            for (int markerId = 0;
                 markerId <= 3;
                 markerId++) {
                if (!foundRequired[markerId]) {
                    missingIds.add(
                            markerId
                    );
                }
            }

            boolean allRequiredFound =
                    foundRequired[0]
                            && foundRequired[1]
                            && foundRequired[2]
                            && foundRequired[3];

            Log.d(
                    "LiveAruco",
                    "Mat="
                            + cameraFrame.cols()
                            + "x"
                            + cameraFrame.rows()
                            + "; channels="
                            + cameraFrame.channels()
                            + "; detectedIds="
                            + detectedIds
                            + "; missing="
                            + missingIds
                            + "; rejected="
                            + rejectedCandidates.size()
                            + "; ready="
                            + allRequiredFound
            );

            return new LiveArucoDetectionResult(
                    detectedIds,
                    missingIds,
                    markers,
                    rejectedCandidates,
                    allRequiredFound
            );

        } catch (Exception detectionError) {
            Log.e(
                    "LiveAruco",
                    "Élő ArUco-detektálási hiba.",
                    detectionError
            );

            return LiveArucoDetectionResult.error(
                    detectionError
                            .getMessage()
            );

        } finally {
            ids.release();

            for (Mat corner
                    : detectedCornerMats) {
                if (corner != null) {
                    corner.release();
                }
            }

            for (Mat corner
                    : rejectedCornerMats) {
                if (corner != null) {
                    corner.release();
                }
            }

            detectedCornerMats.clear();
            rejectedCornerMats.clear();
        }
    }

    private Point[] readLiveArucoCorners(
            Mat cornerMat
    ) {
        if (cornerMat == null
                || cornerMat.empty()) {
            return null;
        }

        Point[] result =
                new Point[4];

        for (int index = 0;
             index < 4;
             index++) {
            double[] coordinate =
                    cornerMat.get(
                            0,
                            index
                    );

            if (coordinate == null
                    || coordinate.length < 2) {
                return null;
            }

            result[index] =
                    new Point(
                            coordinate[0],
                            coordinate[1]
                    );
        }

        return result;
    }

    private void resetLiveArucoDiagnosticSession() {
        synchronized (liveArucoDiagnosticLock) {
            liveArucoDiagnosticHistory.clear();

            liveArucoDiagnosticStartedElapsed =
                    SystemClock.elapsedRealtime();

            liveArucoDiagnosticSaved =
                    false;

            liveArucoDiagnosticSaving =
                    false;

            liveArucoEverFoundAll =
                    false;

            liveArucoDiagnosticSessionName =
                    new SimpleDateFormat(
                            "yyyyMMdd_HHmmss_SSS",
                            Locale.ROOT
                    ).format(
                            new Date()
                    );
        }
    }

    private void recordLiveArucoDiagnosticFrame(
            long elapsedRealtime,
            ImageProxy image,
            Mat cameraFrame,
            LiveArucoDetectionResult result
    ) {
        if (result == null) {
            return;
        }

        synchronized (liveArucoDiagnosticLock) {
            if (liveArucoDiagnosticStartedElapsed <= 0L) {
                liveArucoDiagnosticStartedElapsed =
                        elapsedRealtime;
            }

            if (result.allRequiredFound) {
                liveArucoEverFoundAll =
                        true;
            }

            long sinceStart =
                    Math.max(
                            0L,
                            elapsedRealtime
                                    - liveArucoDiagnosticStartedElapsed
                    );

            liveArucoDiagnosticHistory.add(
                    new LiveArucoFrameHistory(
                            sinceStart,
                            image.getWidth(),
                            image.getHeight(),
                            image.getImageInfo()
                                    .getRotationDegrees(),
                            cameraFrame.cols(),
                            cameraFrame.rows(),
                            cameraFrame.channels(),
                            result.detectedIds,
                            result.missingIds,
                            result.rejectedCandidates.size(),
                            result.errorMessage
                    )
            );

            while (liveArucoDiagnosticHistory.size()
                    > LIVE_ARUCO_DIAGNOSTIC_HISTORY_LIMIT) {
                liveArucoDiagnosticHistory.remove(
                        0
                );
            }
        }
    }

    private void maybeSaveLiveArucoDiagnostic(
            long elapsedRealtime,
            ImageProxy image,
            Mat cameraFrame,
            LiveArucoDetectionResult detectionResult
    ) {
        if (cameraFrame == null
                || cameraFrame.empty()
                || detectionResult == null) {
            return;
        }

        String sessionName;

        synchronized (liveArucoDiagnosticLock) {
            long elapsedSinceStart =
                    liveArucoDiagnosticStartedElapsed <= 0L
                            ? 0L
                            : elapsedRealtime
                              - liveArucoDiagnosticStartedElapsed;

            boolean shouldSave =
                    !liveArucoDiagnosticSaved
                            && !liveArucoDiagnosticSaving
                            && !liveArucoEverFoundAll
                            && !detectionResult.allRequiredFound
                            && elapsedSinceStart
                            >= LIVE_ARUCO_DIAGNOSTIC_DELAY_MS;

            if (!shouldSave) {
                return;
            }

            liveArucoDiagnosticSaving =
                    true;

            sessionName =
                    liveArucoDiagnosticSessionName;

            if (sessionName == null
                    || sessionName.trim().isEmpty()) {
                sessionName =
                        new SimpleDateFormat(
                                "yyyyMMdd_HHmmss_SSS",
                                Locale.ROOT
                        ).format(
                                new Date()
                        );

                liveArucoDiagnosticSessionName =
                        sessionName;
            }
        }

        boolean savedSuccessfully =
                false;

        Mat grayMat =
                null;

        Mat detectedOverlayMat =
                null;

        Mat rejectedOverlayMat =
                null;

        Bitmap fullFrameBitmap =
                null;

        Bitmap grayBitmap =
                null;

        Bitmap detectedOverlayBitmap =
                null;

        Bitmap rejectedOverlayBitmap =
                null;

        try {
            Context context =
                    getContext();

            if (context == null) {
                throw new IOException(
                        "Az alkalmazáskörnyezet nem érhető el."
                );
            }

            Context appContext =
                    context.getApplicationContext();

            fullFrameBitmap =
                    matToLiveArucoDebugBitmap(
                            cameraFrame
                    );

            grayMat =
                    createLiveArucoGrayscale(
                            cameraFrame
                    );

            detectedOverlayMat =
                    createLiveArucoDetectedOverlay(
                            cameraFrame,
                            detectionResult
                    );

            rejectedOverlayMat =
                    createLiveArucoRejectedOverlay(
                            cameraFrame,
                            detectionResult
                    );

            grayBitmap =
                    matToLiveArucoDebugBitmap(
                            grayMat
                    );

            detectedOverlayBitmap =
                    matToLiveArucoDebugBitmap(
                            detectedOverlayMat
                    );

            rejectedOverlayBitmap =
                    matToLiveArucoDebugBitmap(
                            rejectedOverlayMat
                    );

            saveLiveArucoDiagnosticBitmap(
                    appContext,
                    sessionName,
                    "01_live_full_frame.png",
                    fullFrameBitmap
            );

            saveLiveArucoDiagnosticBitmap(
                    appContext,
                    sessionName,
                    "02_live_grayscale.png",
                    grayBitmap
            );

            saveLiveArucoDiagnosticBitmap(
                    appContext,
                    sessionName,
                    "03_detected_markers.png",
                    detectedOverlayBitmap
            );

            saveLiveArucoDiagnosticBitmap(
                    appContext,
                    sessionName,
                    "04_rejected_candidates.png",
                    rejectedOverlayBitmap
            );

            JSONObject diagnosticJson =
                    createLiveArucoDiagnosticJson(
                            sessionName,
                            image,
                            cameraFrame,
                            detectionResult,
                            elapsedRealtime
                    );

            saveLiveArucoDiagnosticText(
                    appContext,
                    sessionName,
                    "aruco_live_diagnostic.json",
                    diagnosticJson.toString(2)
            );

            savedSuccessfully =
                    true;

            String savedPath =
                    "Download/MedScan/Debug/Aruco/"
                            + sessionName;

            Log.d(
                    "LiveAruco",
                    "Élő ArUco-diagnosztika elmentve: "
                            + savedPath
            );

            String finalSavedPath =
                    savedPath;

            handler.post(() -> {
                if (!isAdded()
                        || tvResultName == null
                        || !BuildConfig.DEBUG) {
                    return;
                }

                tvResultName.setText(
                        "Az ArUco-diagnosztika elmentve:\n"
                                + finalSavedPath
                );
            });

        } catch (Exception saveError) {
            Log.e(
                    "LiveAruco",
                    "Az élő ArUco-diagnosztika mentése sikertelen.",
                    saveError
            );

            handler.post(() -> {
                if (!isAdded()
                        || tvResultName == null
                        || !BuildConfig.DEBUG) {
                    return;
                }

                showError(
                        "Az ArUco-diagnosztika mentése nem sikerült.",
                        saveError
                );
            });

        } finally {
            if (fullFrameBitmap != null
                    && !fullFrameBitmap.isRecycled()) {
                fullFrameBitmap.recycle();
            }

            if (grayBitmap != null
                    && !grayBitmap.isRecycled()) {
                grayBitmap.recycle();
            }

            if (detectedOverlayBitmap != null
                    && !detectedOverlayBitmap.isRecycled()) {
                detectedOverlayBitmap.recycle();
            }

            if (rejectedOverlayBitmap != null
                    && !rejectedOverlayBitmap.isRecycled()) {
                rejectedOverlayBitmap.recycle();
            }

            if (grayMat != null) {
                grayMat.release();
            }

            if (detectedOverlayMat != null) {
                detectedOverlayMat.release();
            }

            if (rejectedOverlayMat != null) {
                rejectedOverlayMat.release();
            }

            synchronized (liveArucoDiagnosticLock) {
                liveArucoDiagnosticSaving =
                        false;

                liveArucoDiagnosticSaved =
                        savedSuccessfully;

                if (!savedSuccessfully) {
                    liveArucoDiagnosticStartedElapsed =
                            elapsedRealtime
                                    - LIVE_ARUCO_DIAGNOSTIC_DELAY_MS
                                    + 5000L;
                }
            }
        }
    }

    private Mat createLiveArucoGrayscale(
            Mat source
    ) {
        Mat gray =
                new Mat();

        if (source.channels() == 4) {
            Imgproc.cvtColor(
                    source,
                    gray,
                    Imgproc.COLOR_RGBA2GRAY
            );
        } else if (source.channels() == 3) {
            Imgproc.cvtColor(
                    source,
                    gray,
                    Imgproc.COLOR_RGB2GRAY
            );
        } else {
            source.copyTo(
                    gray
            );
        }

        return gray;
    }

    private Mat createLiveArucoColorOverlayBase(
            Mat source
    ) {
        Mat overlay =
                new Mat();

        if (source.channels() == 1) {
            Imgproc.cvtColor(
                    source,
                    overlay,
                    Imgproc.COLOR_GRAY2RGBA
            );
        } else if (source.channels() == 3) {
            Imgproc.cvtColor(
                    source,
                    overlay,
                    Imgproc.COLOR_RGB2RGBA
            );
        } else {
            source.copyTo(
                    overlay
            );
        }

        return overlay;
    }

    private Mat createLiveArucoDetectedOverlay(
            Mat source,
            LiveArucoDetectionResult result
    ) {
        Mat overlay =
                createLiveArucoColorOverlayBase(
                        source
                );

        Scalar green =
                new Scalar(
                        0,
                        255,
                        0,
                        255
                );

        for (LiveArucoMarker marker
                : result.markers) {
            drawLiveArucoQuadrilateral(
                    overlay,
                    marker.corners,
                    green,
                    4
            );

            Point center =
                    calculateLiveArucoCenter(
                            marker.corners
                    );

            Imgproc.putText(
                    overlay,
                    "ID=" + marker.id,
                    center,
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    green,
                    3
            );
        }

        return overlay;
    }

    private Mat createLiveArucoRejectedOverlay(
            Mat source,
            LiveArucoDetectionResult result
    ) {
        Mat overlay =
                createLiveArucoColorOverlayBase(
                        source
                );

        Scalar green =
                new Scalar(
                        0,
                        255,
                        0,
                        255
                );

        Scalar red =
                new Scalar(
                        255,
                        0,
                        0,
                        255
                );

        for (LiveArucoMarker marker
                : result.markers) {
            drawLiveArucoQuadrilateral(
                    overlay,
                    marker.corners,
                    green,
                    3
            );
        }

        for (Point[] rejectedCandidate
                : result.rejectedCandidates) {
            drawLiveArucoQuadrilateral(
                    overlay,
                    rejectedCandidate,
                    red,
                    4
            );
        }

        Imgproc.putText(
                overlay,
                "Rejected="
                        + result.rejectedCandidates.size(),
                new Point(
                        24,
                        48
                ),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0,
                red,
                3
        );

        return overlay;
    }

    private void drawLiveArucoQuadrilateral(
            Mat image,
            Point[] corners,
            Scalar color,
            int thickness
    ) {
        if (corners == null
                || corners.length < 4) {
            return;
        }

        for (int index = 0;
             index < 4;
             index++) {
            Point start =
                    corners[index];

            Point end =
                    corners[
                            (index + 1) % 4
                            ];

            Imgproc.line(
                    image,
                    start,
                    end,
                    color,
                    thickness
            );
        }
    }

    private Point calculateLiveArucoCenter(
            Point[] corners
    ) {
        double sumX =
                0.0;

        double sumY =
                0.0;

        for (Point corner
                : corners) {
            sumX += corner.x;
            sumY += corner.y;
        }

        return new Point(
                sumX / corners.length,
                sumY / corners.length
        );
    }

    private Bitmap matToLiveArucoDebugBitmap(
            Mat mat
    ) throws Exception {
        if (mat == null
                || mat.empty()) {
            throw new Exception(
                    "A diagnosztikai OpenCV-kép üres."
            );
        }

        Mat bitmapCompatibleMat =
                null;

        try {
            if (mat.channels() == 1) {
                bitmapCompatibleMat =
                        new Mat();

                Imgproc.cvtColor(
                        mat,
                        bitmapCompatibleMat,
                        Imgproc.COLOR_GRAY2RGBA
                );

            } else if (mat.channels() == 3) {
                bitmapCompatibleMat =
                        new Mat();

                Imgproc.cvtColor(
                        mat,
                        bitmapCompatibleMat,
                        Imgproc.COLOR_RGB2RGBA
                );

            } else {
                bitmapCompatibleMat =
                        mat;
            }

            Bitmap bitmap =
                    Bitmap.createBitmap(
                            bitmapCompatibleMat.cols(),
                            bitmapCompatibleMat.rows(),
                            Bitmap.Config.ARGB_8888
                    );

            Utils.matToBitmap(
                    bitmapCompatibleMat,
                    bitmap
            );

            return bitmap;

        } finally {
            if (bitmapCompatibleMat != null
                    && bitmapCompatibleMat != mat) {
                bitmapCompatibleMat.release();
            }
        }
    }

    private JSONObject createLiveArucoDiagnosticJson(
            String sessionName,
            ImageProxy image,
            Mat cameraFrame,
            LiveArucoDetectionResult result,
            long elapsedRealtime
    ) throws Exception {
        JSONObject root =
                new JSONObject();

        root.put(
                "diagnostic_type",
                "live_camera_aruco_detection"
        );

        root.put(
                "reason",
                "A kamera legalább 8 másodpercig egyszer sem találta meg "
                        + "egyszerre a 0, 1, 2 és 3 ArUco-markert."
        );

        root.put(
                "session",
                sessionName
        );

        root.put(
                "saved_at",
                new SimpleDateFormat(
                        "yyyy-MM-dd HH:mm:ss.SSS",
                        Locale.ROOT
                ).format(
                        new Date()
                )
        );

        JSONObject device =
                new JSONObject();

        device.put(
                "manufacturer",
                Build.MANUFACTURER
        );

        device.put(
                "brand",
                Build.BRAND
        );

        device.put(
                "model",
                Build.MODEL
        );

        device.put(
                "device",
                Build.DEVICE
        );

        device.put(
                "product",
                Build.PRODUCT
        );

        device.put(
                "android_release",
                Build.VERSION.RELEASE
        );

        device.put(
                "android_sdk",
                Build.VERSION.SDK_INT
        );

        device.put(
                "opencv_version",
                Core.VERSION
        );

        root.put(
                "device",
                device
        );

        JSONObject frame =
                new JSONObject();

        frame.put(
                "image_proxy_width",
                image.getWidth()
        );

        frame.put(
                "image_proxy_height",
                image.getHeight()
        );

        frame.put(
                "image_format",
                image.getFormat()
        );

        frame.put(
                "rotation_degrees",
                image.getImageInfo()
                        .getRotationDegrees()
        );

        Rect cropRect =
                image.getCropRect();

        frame.put(
                "crop_rect",
                new JSONArray()
                        .put(cropRect.left)
                        .put(cropRect.top)
                        .put(cropRect.right)
                        .put(cropRect.bottom)
        );

        frame.put(
                "crop_rect_applied_to_live_detection",
                false
        );

        frame.put(
                "live_detection_uses_full_frame",
                true
        );

        frame.put(
                "live_detection_input",
                "Y_PLANE_STRIDE_SAFE_CV_8UC1"
        );

        frame.put(
                "nv21_jpeg_used_for_live_detection",
                false
        );

        frame.put(
                "opencv_width",
                cameraFrame.cols()
        );

        frame.put(
                "opencv_height",
                cameraFrame.rows()
        );

        frame.put(
                "opencv_channels",
                cameraFrame.channels()
        );

        JSONArray planes =
                new JSONArray();

        ImageProxy.PlaneProxy[] planeArray =
                image.getPlanes();

        for (int index = 0;
             index < planeArray.length;
             index++) {
            ImageProxy.PlaneProxy plane =
                    planeArray[index];

            planes.put(
                    new JSONObject()
                            .put(
                                    "index",
                                    index
                            )
                            .put(
                                    "row_stride",
                                    plane.getRowStride()
                            )
                            .put(
                                    "pixel_stride",
                                    plane.getPixelStride()
                            )
                            .put(
                                    "remaining_bytes",
                                    plane.getBuffer()
                                            .remaining()
                            )
            );
        }

        frame.put(
                "planes",
                planes
        );

        root.put(
                "frame",
                frame
        );

        JSONObject state =
                new JSONObject();

        state.put(
                "phone_level",
                isPhoneLevel
        );

        state.put(
                "camera_started",
                cameraStarted
        );

        state.put(
                "camera_calibration_available",
                cameraCalibration != null
        );

        state.put(
                "height_measurement_valid",
                isHeightMeasurementValid
        );

        if (!Double.isNaN(
                filteredCameraHeightCm
        )) {
            state.put(
                    "camera_height_cm",
                    filteredCameraHeightCm
            );
        }

        state.put(
                "required_detected_ids",
                toJsonIntegerArray(
                        result.detectedIds
                )
        );

        state.put(
                "missing_ids",
                toJsonIntegerArray(
                        result.missingIds
                )
        );

        state.put(
                "rejected_candidates_count",
                result.rejectedCandidates.size()
        );

        state.put(
                "all_required_found",
                result.allRequiredFound
        );

        if (result.errorMessage != null
                && !result.errorMessage.trim().isEmpty()) {
            state.put(
                    "detector_error",
                    result.errorMessage
            );
        }

        root.put(
                "last_result",
                state
        );

        JSONArray markersJson =
                new JSONArray();

        for (LiveArucoMarker marker
                : result.markers) {
            JSONObject markerJson =
                    new JSONObject();

            markerJson.put(
                    "id",
                    marker.id
            );

            markerJson.put(
                    "corners",
                    pointsToJson(
                            marker.corners
                    )
            );

            Point center =
                    calculateLiveArucoCenter(
                            marker.corners
                    );

            markerJson.put(
                    "center",
                    new JSONArray()
                            .put(center.x)
                            .put(center.y)
            );

            markerJson.put(
                    "average_side_px",
                    calculateLiveArucoAverageSide(
                            marker.corners
                    )
            );

            markersJson.put(
                    markerJson
            );
        }

        root.put(
                "detected_markers",
                markersJson
        );

        JSONArray rejectedJson =
                new JSONArray();

        for (Point[] rejected
                : result.rejectedCandidates) {
            rejectedJson.put(
                    pointsToJson(
                            rejected
                    )
            );
        }

        root.put(
                "rejected_candidates",
                rejectedJson
        );

        JSONArray historyJson =
                new JSONArray();

        synchronized (liveArucoDiagnosticLock) {
            for (LiveArucoFrameHistory history
                    : liveArucoDiagnosticHistory) {
                historyJson.put(
                        history.toJson()
                );
            }
        }

        root.put(
                "detection_history",
                historyJson
        );

        long elapsedSinceStart;

        synchronized (liveArucoDiagnosticLock) {
            elapsedSinceStart =
                    liveArucoDiagnosticStartedElapsed <= 0L
                            ? 0L
                            : elapsedRealtime
                              - liveArucoDiagnosticStartedElapsed;
        }

        root.put(
                "elapsed_since_camera_start_ms",
                elapsedSinceStart
        );

        return root;
    }

    private JSONArray toJsonIntegerArray(
            List<Integer> values
    ) {
        JSONArray array =
                new JSONArray();

        if (values == null) {
            return array;
        }

        for (Integer value
                : values) {
            array.put(
                    value
            );
        }

        return array;
    }

    private JSONArray pointsToJson(
            Point[] points
    ) throws JSONException {
        JSONArray result =
                new JSONArray();

        if (points == null) {
            return result;
        }

        for (Point point : points) {
            if (point == null) {
                continue;
            }

            result.put(
                    new JSONArray()
                            .put(point.x)
                            .put(point.y)
            );
        }

        return result;
    }

    private double calculateLiveArucoAverageSide(
            Point[] corners
    ) {
        if (corners == null
                || corners.length < 4) {
            return 0.0;
        }

        double total =
                0.0;

        for (int index = 0;
             index < 4;
             index++) {
            Point first =
                    corners[index];

            Point second =
                    corners[
                            (index + 1) % 4
                            ];

            double dx =
                    first.x - second.x;

            double dy =
                    first.y - second.y;

            total +=
                    Math.sqrt(
                            dx * dx + dy * dy
                    );
        }

        return total / 4.0;
    }

    private void saveLiveArucoDiagnosticBitmap(
            Context context,
            String sessionName,
            String filename,
            Bitmap bitmap
    ) throws Exception {
        if (bitmap == null
                || bitmap.isRecycled()) {
            throw new IOException(
                    "A mentendő ArUco-diagnosztikai kép hibás."
            );
        }

        if (Build.VERSION.SDK_INT
                >= Build.VERSION_CODES.Q) {
            saveLiveArucoBitmapWithMediaStore(
                    context,
                    sessionName,
                    filename,
                    bitmap
            );
        } else {
            saveLiveArucoBitmapLegacy(
                    context,
                    sessionName,
                    filename,
                    bitmap
            );
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private void saveLiveArucoBitmapWithMediaStore(
            Context context,
            String sessionName,
            String filename,
            Bitmap bitmap
    ) throws Exception {
        ContentValues values =
                new ContentValues();

        values.put(
                MediaStore.MediaColumns.DISPLAY_NAME,
                filename
        );

        values.put(
                MediaStore.MediaColumns.MIME_TYPE,
                "image/png"
        );

        values.put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOWNLOADS
                        + "/MedScan/Debug/Aruco/"
                        + sessionName
        );

        values.put(
                MediaStore.MediaColumns.IS_PENDING,
                1
        );

        Uri uri =
                context.getContentResolver()
                        .insert(
                                MediaStore.Downloads
                                        .EXTERNAL_CONTENT_URI,
                                values
                        );

        if (uri == null) {
            throw new IOException(
                    "A MediaStore nem hozta létre a fájlt: "
                            + filename
            );
        }

        boolean completed =
                false;

        try (OutputStream outputStream =
                     context.getContentResolver()
                             .openOutputStream(
                                     uri
                             )) {
            if (outputStream == null) {
                throw new IOException(
                        "Nem nyitható meg a fájl: "
                                + filename
                );
            }

            boolean compressed =
                    bitmap.compress(
                            Bitmap.CompressFormat.PNG,
                            100,
                            outputStream
                    );

            if (!compressed) {
                throw new IOException(
                        "A PNG-tömörítés sikertelen: "
                                + filename
                );
            }

            outputStream.flush();
            completed = true;

        } finally {
            if (!completed) {
                context.getContentResolver()
                        .delete(
                                uri,
                                null,
                                null
                        );
            }
        }

        ContentValues finished =
                new ContentValues();

        finished.put(
                MediaStore.MediaColumns.IS_PENDING,
                0
        );

        context.getContentResolver()
                .update(
                        uri,
                        finished,
                        null,
                        null
                );
    }

    private void saveLiveArucoBitmapLegacy(
            Context context,
            String sessionName,
            String filename,
            Bitmap bitmap
    ) throws Exception {
        File directory =
                createLegacyLiveArucoDirectory(
                        context,
                        sessionName
                );

        File target =
                new File(
                        directory,
                        filename
                );

        try (OutputStream outputStream =
                     new FileOutputStream(
                             target
                     )) {
            boolean compressed =
                    bitmap.compress(
                            Bitmap.CompressFormat.PNG,
                            100,
                            outputStream
                    );

            if (!compressed) {
                throw new IOException(
                        "A PNG-tömörítés sikertelen: "
                                + filename
                );
            }

            outputStream.flush();
        }
    }

    private void saveLiveArucoDiagnosticText(
            Context context,
            String sessionName,
            String filename,
            String content
    ) throws Exception {
        byte[] bytes =
                content.getBytes(
                        StandardCharsets.UTF_8
                );

        if (Build.VERSION.SDK_INT
                >= Build.VERSION_CODES.Q) {
            saveLiveArucoTextWithMediaStore(
                    context,
                    sessionName,
                    filename,
                    bytes
            );
        } else {
            File directory =
                    createLegacyLiveArucoDirectory(
                            context,
                            sessionName
                    );

            File target =
                    new File(
                            directory,
                            filename
                    );

            try (OutputStream outputStream =
                         new FileOutputStream(
                                 target
                         )) {
                outputStream.write(
                        bytes
                );

                outputStream.flush();
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private void saveLiveArucoTextWithMediaStore(
            Context context,
            String sessionName,
            String filename,
            byte[] bytes
    ) throws Exception {
        ContentValues values =
                new ContentValues();

        values.put(
                MediaStore.MediaColumns.DISPLAY_NAME,
                filename
        );

        values.put(
                MediaStore.MediaColumns.MIME_TYPE,
                "application/json"
        );

        values.put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOWNLOADS
                        + "/MedScan/Debug/Aruco/"
                        + sessionName
        );

        values.put(
                MediaStore.MediaColumns.IS_PENDING,
                1
        );

        Uri uri =
                context.getContentResolver()
                        .insert(
                                MediaStore.Downloads
                                        .EXTERNAL_CONTENT_URI,
                                values
                        );

        if (uri == null) {
            throw new IOException(
                    "A MediaStore nem hozta létre a fájlt: "
                            + filename
            );
        }

        boolean completed =
                false;

        try (OutputStream outputStream =
                     context.getContentResolver()
                             .openOutputStream(
                                     uri
                             )) {
            if (outputStream == null) {
                throw new IOException(
                        "Nem nyitható meg a fájl: "
                                + filename
                );
            }

            outputStream.write(
                    bytes
            );

            outputStream.flush();
            completed = true;

        } finally {
            if (!completed) {
                context.getContentResolver()
                        .delete(
                                uri,
                                null,
                                null
                        );
            }
        }

        ContentValues finished =
                new ContentValues();

        finished.put(
                MediaStore.MediaColumns.IS_PENDING,
                0
        );

        context.getContentResolver()
                .update(
                        uri,
                        finished,
                        null,
                        null
                );
    }

    private File createLegacyLiveArucoDirectory(
            Context context,
            String sessionName
    ) throws Exception {
        File downloadsRoot =
                context.getExternalFilesDir(
                        Environment.DIRECTORY_DOWNLOADS
                );

        if (downloadsRoot == null) {
            throw new IOException(
                    "A külső Download mappa nem érhető el."
            );
        }

        File directory =
                new File(
                        downloadsRoot,
                        "MedScan/Debug/Aruco/"
                                + sessionName
                );

        if (!directory.exists()
                && !directory.mkdirs()) {
            throw new IOException(
                    "Az ArUco diagnosztikai mappa nem hozható létre."
            );
        }

        return directory;
    }

    private void updateCameraMeasurement(
            ArUcoHeightEstimator.HeightMeasurement measurement,
            boolean allMarkersFound
    ) {

        lastArucoState =
                allMarkersFound;

        if (measurement == null) {
            isHeightMeasurementValid = false;
            filteredCameraHeightCm = Double.NaN;
            lastHeightMeasurementMessage = "Nincs mérési eredmény.";
            refreshCameraReadyUi();
            return;
        }

        lastHeightMeasurementMessage =
                measurement.getMessage();

        if (measurement.isValid()) {
            double rawHeightCm = measurement.getHeightCm();

            if (Double.isNaN(filteredCameraHeightCm)) {
                filteredCameraHeightCm = rawHeightCm;
            } else {
                filteredCameraHeightCm =
                        filteredCameraHeightCm
                                + HEIGHT_SMOOTHING_ALPHA
                                * (rawHeightCm - filteredCameraHeightCm);
            }

            isHeightMeasurementValid = true;
            lastHeightMeasurementElapsed = SystemClock.elapsedRealtime();

            Log.d(
                    "PillHeight",
                    "Magasság="
                            + formatHeightCm(filteredCameraHeightCm)
                            + " cm; szélesség alapján="
                            + formatHeightCm(
                            measurement.getWidthBasedHeightCm()
                    )
                            + " cm; magasság alapján="
                            + formatHeightCm(
                            measurement.getHeightBasedHeightCm()
                    )
                            + " cm; kalibráció="
                            + measurement.getCalibrationSource()
            );
        } else {
            isHeightMeasurementValid = false;
            filteredCameraHeightCm = Double.NaN;
            lastHeightMeasurementElapsed = 0L;
        }

        refreshCameraReadyUi();
    }

    private void refreshCameraReadyUi() {
        if (!isAdded()) return;

        boolean hasFreshHeight = hasFreshHeightMeasurement();
        boolean isWithinHeightLimit =
                hasFreshHeight
                        && filteredCameraHeightCm <= MAX_CAMERA_HEIGHT_CM;

        boolean isMarkerFrameReady =
                cameraStarted
                        && !isTakingPhoto
                        && !isProcessing
                        && isPhoneLevel
                        && lastArucoState;

        boolean isReadyToCapture =
                isMarkerFrameReady
                        && hasFreshHeight
                        && isWithinHeightLimit;

        if (btnTakePhotoFull != null) {
            btnTakePhotoFull.setEnabled(
                    isReadyToCapture
            );
        }

        if (tvCameraStatus != null) {
            if (cameraCalibration == null) {
                String technicalMessage =
                        lastHeightMeasurementMessage == null
                                || lastHeightMeasurementMessage.trim().isEmpty()
                                ? "A kamera kalibrációs adatai nem érhetők el."
                                : lastHeightMeasurementMessage;

                tvCameraStatus.setText(
                        BuildConfig.DEBUG
                                ? "A kamera beállítása nem sikerült.\n"
                                  + "[DEBUG] "
                                  + technicalMessage
                                : "A kamera beállítása nem sikerült. "
                                  + "Zárd be, majd nyisd meg újra a kamerát."
                );

            } else if (!isPhoneLevel) {
                tvCameraStatus.setText(
                        "Igazítsd vízszintbe a telefont."
                );

            } else if (!lastArucoState) {
                tvCameraStatus.setText(
                        BuildConfig.DEBUG
                                ? "Keresem a 0, 1, 2 és 3 "
                                  + "ArUco-jelölést..."
                                : "Keresem a lap négy "
                                  + "sarokjelölését..."
                );

            } else if (!hasFreshHeight) {
                String technicalMessage =
                        lastHeightMeasurementMessage == null
                                || lastHeightMeasurementMessage.trim().isEmpty()
                                ? "A magasság még nem mérhető."
                                : lastHeightMeasurementMessage;

                tvCameraStatus.setText(
                        BuildConfig.DEBUG
                                ? "A távolság mérése folyamatban van.\n"
                                  + "[DEBUG] "
                                  + technicalMessage
                                : "A távolság mérése folyamatban van. "
                                  + "Tartsd stabilan a készüléket és "
                                  + "mutasd a teljes lapot."
                );
            } else if (!isWithinHeightLimit) {
                tvCameraStatus.setText(
                        "Magasság: "
                                + formatHeightCm(filteredCameraHeightCm)
                                + " cm / maximum: "
                                + formatHeightCm(MAX_CAMERA_HEIGHT_CM)
                                + " cm. Vidd közelebb a telefont."
                );
            } else {
                tvCameraStatus.setText(
                        "Magasság: "
                                + formatHeightCm(filteredCameraHeightCm)
                                + " cm / maximum: "
                                + formatHeightCm(MAX_CAMERA_HEIGHT_CM)
                                + " cm. Készíthető fotó."
                );
            }
        }

        int color =
                isMarkerFrameReady
                        ? Color.GREEN
                        : Color.RED;

        View root = getView();
        if (root == null) return;

        View overlay = root.findViewById(R.id.cornerGuideOverlay);

        if (overlay instanceof ViewGroup
                && ((ViewGroup) overlay).getChildCount() > 0
                && ((ViewGroup) overlay).getChildAt(0) instanceof ViewGroup) {

            ViewGroup container =
                    (ViewGroup) ((ViewGroup) overlay).getChildAt(0);

            for (int i = 0; i < container.getChildCount(); i++) {
                container.getChildAt(i).setBackgroundColor(color);
            }
        }
    }

    private boolean hasFreshHeightMeasurement() {
        return isHeightMeasurementValid
                && !Double.isNaN(filteredCameraHeightCm)
                && SystemClock.elapsedRealtime()
                - lastHeightMeasurementElapsed
                <= HEIGHT_MEASUREMENT_MAX_AGE_MS;
    }

    private void resetHeightMeasurementState() {
        lastArucoState = false;
        isHeightMeasurementValid = false;
        filteredCameraHeightCm = Double.NaN;
        lastHeightMeasurementElapsed = 0L;
        lastHeightMeasurementMessage = "";
        lastArucoCheckTime = 0L;
    }

    private String formatHeightCm(double value) {
        return String.format(
                Locale.getDefault(),
                "%.1f",
                value
        );
    }

    private void takePhotoIfReady() {

        if (isTakingPhoto) {

            return;

        }

        if (!isPhoneLevel) {
            showError(
                    "Nem készíthető kép! Tartsd vízszintesen a telefont."
            );
            refreshCameraReadyUi();
            return;
        }

        if (!lastArucoState) {
            showError(
                    "A fénykép még nem készíthető el. "
                            + "Mutasd a lap mind a négy "
                            + "sarokjelölését."
            );
            refreshCameraReadyUi();
            return;
        }

        if (!hasFreshHeightMeasurement()) {
            showError(
                    "A fénykép még nem készíthető el. "
                            + "Tartsd stabilan a készüléket, és "
                            + "várd meg a távolságmérést."
            );
            refreshCameraReadyUi();
            return;
        }

        if (filteredCameraHeightCm > MAX_CAMERA_HEIGHT_CM) {
            showError(
                    "Túl magasan van a telefon: "
                            + formatHeightCm(filteredCameraHeightCm)
                            + " cm. A maximum "
                            + formatHeightCm(MAX_CAMERA_HEIGHT_CM)
                            + " cm."
            );
            refreshCameraReadyUi();
            return;
        }

        if (imageCapture == null) {

            showError(
                    "A kamera még nem áll készen. "
                            + "Várj egy pillanatot, majd próbáld újra."
            );

            return;

        }

        capturedCameraHeightCm = filteredCameraHeightCm;
        isTakingPhoto = true;

        if (btnTakePhotoFull != null) {

            btnTakePhotoFull.setEnabled(false);

        }

        if (btnCloseCamera != null) {

            btnCloseCamera.setEnabled(false);

        }

        if (btnOpenCamera != null) {

            btnOpenCamera.setEnabled(false);

        }

        if (tvCameraStatus != null) {

            tvCameraStatus.setText("Fénykép készítése...");

        }

        tvResultName.setText("Fénykép készítése...");

        imageCapture.takePicture(

                executor,

                new ImageCapture.OnImageCapturedCallback() {

                    @Override

                    public void onCaptureSuccess(@NonNull ImageProxy image) {

                        Log.d("PillScan", "Kép sikeresen befogva!");

                        Bitmap rotatedBitmap;

                        try {

                            Bitmap bitmap = imageProxyToBitmap(image);

                            int rotationDegrees =
                                    image.getImageInfo()
                                            .getRotationDegrees();

                            Bitmap correctlyRotatedBitmap =
                                    rotateBitmap(
                                            bitmap,
                                            rotationDegrees
                                    );

                            rotatedBitmap =
                                    normalizeCapturedPillPhoto(
                                            correctlyRotatedBitmap
                                    );

                            Log.d(
                                    "PillScan",
                                    "Bitmap forgatás és tájoláshelyes 4:3 "
                                            + "normalizálás kész: "
                                            + rotatedBitmap.getWidth()
                                            + "x"
                                            + rotatedBitmap.getHeight()
                            );

                        } catch (Exception e) {

                            image.close();

                            Log.e("PillScan", "Hiba a konvertálásban: " + e.getMessage());

                            handler.post(() -> {

                                if (!isAdded() || getView() == null) return;

                                isTakingPhoto = false;

                                refreshCameraReadyUi();

                                if (btnCloseCamera != null) btnCloseCamera.setEnabled(true);

                                if (btnOpenCamera != null) btnOpenCamera.setEnabled(true);

                                showError(
                                        createUserFriendlyProcessingError(e),
                                        e
                                );

                            });

                            return;

                        }

                        image.close();

                        Log.d("PillScan", "ImageProxy bezárva.");

                        Bitmap finalRotatedBitmap = rotatedBitmap;

                        handler.post(() -> {

                            if (!isAdded() || getView() == null) return;

                            Log.d(
                                    "PillScan",
                                    "Kép szerkesztési módba helyezése."
                            );

                            if (fullCameraContainer != null) {
                                fullCameraContainer.setVisibility(
                                        View.GONE
                                );
                            }

                            if (btnCloseCamera != null) {
                                btnCloseCamera.setEnabled(true);
                            }

                            isTakingPhoto = false;
                            cameraStarted = false;
                            imageCapture = null;

                            stopCamera();

                            currentEvaluationFilename = "";

                            preparePillPhotoForEditing(
                                    finalRotatedBitmap
                            );

                        });

                    }

                    @Override

                    public void onError(@NonNull ImageCaptureException exception) {

                        handler.post(() -> {

                            if (!isAdded() || getView() == null) return;

                            isTakingPhoto = false;

                            refreshCameraReadyUi();

                            if (btnCloseCamera != null) {

                                btnCloseCamera.setEnabled(true);

                            }

                            if (btnOpenCamera != null) {

                                btnOpenCamera.setEnabled(true);

                            }

                            showError(
                                    "Nem sikerült elkészíteni a fényképet. "
                                            + "Tartsd stabilan a készüléket, "
                                            + "majd próbáld újra.",
                                    exception
                            );

                        });

                    }

                }

        );

    }

    private void stopCamera() {

        try {

            ProcessCameraProvider.getInstance(requireContext()).addListener(() -> {

                try {

                    ProcessCameraProvider provider = ProcessCameraProvider.getInstance(requireContext()).get();

                    provider.unbindAll();

                } catch (Exception ignored) {

                }

            }, ContextCompat.getMainExecutor(requireContext()));

        } catch (Exception ignored) {

        }

    }

    private Bitmap decodeImageProxyToBitmap(
            @NonNull ImageProxy image
    ) {
        Bitmap decodedBitmap;

        if (image.getFormat() == ImageFormat.JPEG) {
            ByteBuffer buffer =
                    image.getPlanes()[0]
                            .getBuffer()
                            .duplicate();

            buffer.rewind();

            byte[] bytes =
                    new byte[
                            buffer.remaining()
                            ];

            buffer.get(
                    bytes
            );

            decodedBitmap =
                    BitmapFactory.decodeByteArray(
                            bytes,
                            0,
                            bytes.length
                    );

        } else if (image.getFormat()
                == ImageFormat.YUV_420_888) {
            byte[] nv21 =
                    yuv420888ToNv21(
                            image
                    );

            YuvImage yuvImage =
                    new YuvImage(
                            nv21,
                            ImageFormat.NV21,
                            image.getWidth(),
                            image.getHeight(),
                            null
                    );

            try (ByteArrayOutputStream outputStream =
                         new ByteArrayOutputStream()) {
                boolean compressed =
                        yuvImage.compressToJpeg(
                                new Rect(
                                        0,
                                        0,
                                        image.getWidth(),
                                        image.getHeight()
                                ),
                                95,
                                outputStream
                        );

                if (!compressed) {
                    throw new IllegalStateException(
                            "A YUV-kép JPEG-kódolása sikertelen."
                    );
                }

                byte[] imageBytes =
                        outputStream.toByteArray();

                decodedBitmap =
                        BitmapFactory.decodeByteArray(
                                imageBytes,
                                0,
                                imageBytes.length
                        );

            } catch (IOException ioError) {
                throw new IllegalStateException(
                        "A CameraX YUV-kép dekódolása sikertelen.",
                        ioError
                );
            }

        } else {
            throw new IllegalArgumentException(
                    "Nem támogatott CameraX képformátum: "
                            + image.getFormat()
            );
        }

        if (decodedBitmap == null) {
            throw new IllegalStateException(
                    "A CameraX képből nem hozható létre Bitmap."
            );
        }

        return decodedBitmap;
    }

    private byte[] yuv420888ToNv21(
            @NonNull ImageProxy image
    ) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException(
                    "Az NV21-konverzió YUV_420_888 formátumot vár."
            );
        }

        ImageProxy.PlaneProxy[] planes =
                image.getPlanes();

        if (planes == null || planes.length < 3) {
            throw new IllegalStateException(
                    "A YUV-kép nem tartalmaz három síkot."
            );
        }

        int width =
                image.getWidth();

        int height =
                image.getHeight();

        int chromaWidth =
                (width + 1) / 2;

        int chromaHeight =
                (height + 1) / 2;

        byte[] nv21 =
                new byte[
                        width * height
                                + chromaWidth
                                * chromaHeight
                                * 2
                        ];

        copyPlaneToPackedBuffer(
                planes[0],
                width,
                height,
                nv21,
                0,
                1,
                0
        );

        int chromaOffset =
                width * height;

        copyPlaneToPackedBuffer(
                planes[2],
                chromaWidth,
                chromaHeight,
                nv21,
                chromaOffset,
                2,
                0
        );

        copyPlaneToPackedBuffer(
                planes[1],
                chromaWidth,
                chromaHeight,
                nv21,
                chromaOffset,
                2,
                1
        );

        return nv21;
    }

    private void copyPlaneToPackedBuffer(
            ImageProxy.PlaneProxy plane,
            int outputWidth,
            int outputHeight,
            byte[] destination,
            int destinationOffset,
            int destinationPixelStride,
            int destinationComponentOffset
    ) {
        ByteBuffer source =
                plane.getBuffer()
                        .duplicate();

        source.rewind();

        int sourceRowStride =
                plane.getRowStride();

        int sourcePixelStride =
                plane.getPixelStride();

        int sourceLimit =
                source.limit();

        for (int row = 0;
             row < outputHeight;
             row++) {
            int sourceRowStart =
                    row * sourceRowStride;

            for (int column = 0;
                 column < outputWidth;
                 column++) {
                int sourceIndex =
                        sourceRowStart
                                + column
                                * sourcePixelStride;

                if (sourceIndex >= sourceLimit) {
                    throw new IllegalStateException(
                            "A YUV-sík rövidebb a vártnál. "
                                    + "row="
                                    + row
                                    + ", column="
                                    + column
                                    + ", rowStride="
                                    + sourceRowStride
                                    + ", pixelStride="
                                    + sourcePixelStride
                                    + ", limit="
                                    + sourceLimit
                                    + ", szükséges index="
                                    + sourceIndex
                    );
                }

                int destinationIndex =
                        destinationOffset
                                + (
                                row
                                        * outputWidth
                                        + column
                        )
                                * destinationPixelStride
                                + destinationComponentOffset;

                destination[destinationIndex] =
                        source.get(
                                sourceIndex
                        );
            }
        }
    }

    private Bitmap imageProxyToBitmap(
            @NonNull ImageProxy image
    ) {
        Bitmap decodedBitmap =
                decodeImageProxyToBitmap(
                        image
                );

        return applyImageProxyCropRect(
                decodedBitmap,
                image
        );
    }

    private Bitmap imageProxyToBitmapForLiveAruco(
            @NonNull ImageProxy image
    ) {
        Mat grayMat =
                imageProxyToGrayscaleMatForLiveAruco(
                        image
                );

        try {
            return matToLiveArucoDebugBitmap(
                    grayMat
            );
        } catch (Exception conversionError) {
            throw new IllegalStateException(
                    "Az élő ArUco Bitmap nem hozható létre.",
                    conversionError
            );
        } finally {
            grayMat.release();
        }
    }

    private Bitmap applyImageProxyCropRect(
            @NonNull Bitmap source,
            @NonNull ImageProxy image
    ) {
        Rect cropRect = image.getCropRect();

        if (cropRect == null
                || cropRect.width() <= 0
                || cropRect.height() <= 0) {
            return source;
        }

        if (cropRect.left == 0
                && cropRect.top == 0
                && cropRect.right == image.getWidth()
                && cropRect.bottom == image.getHeight()) {
            return source;
        }

        float scaleX =
                source.getWidth()
                        / (float) image.getWidth();

        float scaleY =
                source.getHeight()
                        / (float) image.getHeight();

        int left = clamp(
                Math.round(cropRect.left * scaleX),
                0,
                source.getWidth() - 1
        );

        int top = clamp(
                Math.round(cropRect.top * scaleY),
                0,
                source.getHeight() - 1
        );

        int right = clamp(
                Math.round(cropRect.right * scaleX),
                left + 1,
                source.getWidth()
        );

        int bottom = clamp(
                Math.round(cropRect.bottom * scaleY),
                top + 1,
                source.getHeight()
        );

        Bitmap cropped =
                Bitmap.createBitmap(
                        source,
                        left,
                        top,
                        right - left,
                        bottom - top
                );

        if (cropped != source
                && !source.isRecycled()) {
            source.recycle();
        }

        return cropped;
    }

    private Bitmap normalizeCapturedPillPhoto(
            @NonNull Bitmap source
    ) {
        if (source == null || source.isRecycled()) {
            throw new IllegalArgumentException(
                    "A normalizálandó Bitmap hibás vagy már fel lett szabadítva."
            );
        }

        if (source.getWidth() <= 0
                || source.getHeight() <= 0) {
            throw new IllegalArgumentException(
                    "Érvénytelen forráskép-méret."
            );
        }

        int originalWidth = source.getWidth();
        int originalHeight = source.getHeight();

        Bitmap landscapeBitmap = source;

        if (source.getHeight() > source.getWidth()) {
            Matrix rotationMatrix = new Matrix();
            rotationMatrix.postRotate(90.0f);

            landscapeBitmap =
                    Bitmap.createBitmap(
                            source,
                            0,
                            0,
                            source.getWidth(),
                            source.getHeight(),
                            rotationMatrix,
                            true
                    );

            if (landscapeBitmap != source
                    && !source.isRecycled()) {
                source.recycle();
            }
        }

        Bitmap normalizedBitmap =
                Bitmap.createScaledBitmap(
                        landscapeBitmap,
                        CAMERA_TARGET_WIDTH_PX,
                        CAMERA_TARGET_HEIGHT_PX,
                        true
                );

        if (normalizedBitmap != landscapeBitmap
                && !landscapeBitmap.isRecycled()) {
            landscapeBitmap.recycle();
        }

        Log.d(
                "PillScan",
                "Kötelező fekvő felbontás: "
                        + originalWidth
                        + "x"
                        + originalHeight
                        + " -> "
                        + normalizedBitmap.getWidth()
                        + "x"
                        + normalizedBitmap.getHeight()
        );

        return normalizedBitmap;
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {

        if (rotationDegrees == 0) {

            return bitmap;

        }

        Matrix matrix = new Matrix();

        matrix.postRotate(rotationDegrees);

        return Bitmap.createBitmap(

                bitmap,

                0,

                0,

                bitmap.getWidth(),

                bitmap.getHeight(),

                matrix,

                true

        );

    }

    @Override

    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override

    public void onSensorChanged(SensorEvent event) {

        if (event == null || event.sensor == null || event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) {

            return;

        }

        float rawX = event.values[0];

        float rawY = event.values[1];

        float rawZ = event.values[2];

        if (!hasSensorFilter) {

            filteredX = rawX;

            filteredY = rawY;

            filteredZ = rawZ;

            hasSensorFilter = true;

        } else {

            filteredX = filteredX + SENSOR_SMOOTHING_ALPHA * (rawX - filteredX);

            filteredY = filteredY + SENSOR_SMOOTHING_ALPHA * (rawY - filteredY);

            filteredZ = filteredZ + SENSOR_SMOOTHING_ALPHA * (rawZ - filteredZ);

        }

        double tiltX = Math.toDegrees(Math.atan2(filteredX, filteredZ));

        double tiltY = Math.toDegrees(Math.atan2(filteredY, filteredZ));

        isPhoneLevel =

                Math.abs(tiltX) <= LEVEL_TOLERANCE_DEGREES &&

                        Math.abs(tiltY) <= LEVEL_TOLERANCE_DEGREES &&

                        filteredZ > MIN_FACE_UP_Z;

        updateLevelIndicator(isPhoneLevel);

        if (cameraStarted && !isTakingPhoto) {
            refreshCameraReadyUi();
        }

    }

    @Override

    public void onResume() {

        super.onResume();

        if (sensorManager != null && accelerometer != null) {

            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);

        }

    }

    @Override

    public void onPause() {

        super.onPause();

        if (sensorManager != null) {

            sensorManager.unregisterListener(this);

        }

        if (cameraStarted) {

            stopCamera();

            cameraStarted = false;

            imageCapture = null;
            cameraCalibration = null;
            resetHeightMeasurementState();

        }

        if (fullCameraContainer != null) {

            fullCameraContainer.setVisibility(View.GONE);

        }

        isTakingPhoto = false;

    }

    private Mat createPillMask(
            int width,
            int height,
            List<Point> polygonPoints,
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox
    ) throws Exception {
        if (polygonPoints == null
                || polygonPoints.size() < 3) {
            throw new Exception(
                    "A gyógyszermaszk legalább 3 pontot igényel."
            );
        }

        if (width <= 0 || height <= 0) {
            throw new Exception(
                    "A gyógyszermaszk képmérete hibás."
            );
        }

        List<Point> safePoints =
                new ArrayList<>();

        for (Point point : polygonPoints) {
            if (point == null
                    || Double.isNaN(point.x)
                    || Double.isNaN(point.y)
                    || Double.isInfinite(point.x)
                    || Double.isInfinite(point.y)) {
                continue;
            }

            Point safePoint =
                    new Point(
                            Math.max(
                                    0.0,
                                    Math.min(
                                            width - 1.0,
                                            point.x
                                    )
                            ),
                            Math.max(
                                    0.0,
                                    Math.min(
                                            height - 1.0,
                                            point.y
                                    )
                            )
                    );

            if (safePoints.isEmpty()
                    || pointDistance(
                    safePoints.get(
                            safePoints.size() - 1
                    ),
                    safePoint
            ) >= 0.75) {
                safePoints.add(
                        safePoint
                );
            }
        }

        if (safePoints.size() >= 2
                && pointDistance(
                safePoints.get(0),
                safePoints.get(
                        safePoints.size() - 1
                )
        ) < 0.75) {
            safePoints.remove(
                    safePoints.size() - 1
            );
        }

        if (safePoints.size() < 3) {
            throw new Exception(
                    "A Gemini maszkja nem tartalmaz elég "
                            + "érvényes pontot."
            );
        }

        Mat mask =
                Mat.zeros(
                        height,
                        width,
                        CvType.CV_8UC1
                );

        MatOfPoint originalContour =
                new MatOfPoint();

        MatOfInt hullIndices =
                new MatOfInt();

        MatOfPoint hullContour =
                new MatOfPoint();

        MatOfPoint2f hullContour2f =
                new MatOfPoint2f();

        originalContour.fromList(
                safePoints
        );

        try {
            Imgproc.convexHull(
                    originalContour,
                    hullIndices
            );

            int[] indices =
                    hullIndices.toArray();

            Point[] originalPoints =
                    originalContour.toArray();

            if (indices.length < 3) {
                throw new Exception(
                        "A Gemini pontjaiból nem hozható létre "
                                + "érvényes külső gyógyszerkontúr."
                );
            }

            List<Point> hullPoints =
                    new ArrayList<>(
                            indices.length
                    );

            for (int pointIndex : indices) {
                if (pointIndex >= 0
                        && pointIndex
                        < originalPoints.length) {
                    hullPoints.add(
                            originalPoints[
                                    pointIndex
                                    ]
                    );
                }
            }

            if (hullPoints.size() < 3) {
                throw new Exception(
                        "A külső gyógyszerkontúr túl kevés "
                                + "pontot tartalmaz."
                );
            }

            hullContour.fromList(
                    hullPoints
            );

            hullContour2f.fromList(
                    hullPoints
            );

            double originalArea =
                    Math.abs(
                            Imgproc.contourArea(
                                    originalContour
                            )
                    );

            double hullArea =
                    Math.abs(
                            Imgproc.contourArea(
                                    hullContour
                            )
                    );

            double imageArea =
                    (double) width
                            * height;

            if (hullArea
                    < Math.max(
                    20.0,
                    imageArea * 0.00002
            )) {
                throw new Exception(
                        "A kapott gyógyszermaszk túl kicsi."
                );
            }

            if (hullArea
                    > imageArea * 0.75) {
                throw new Exception(
                        "A kapott gyógyszermaszk túl nagy."
                );
            }

            double perimeter =
                    Imgproc.arcLength(
                            hullContour2f,
                            true
                    );

            double circularity =
                    perimeter <= 0.0
                            ? 0.0
                            : 4.0
                              * Math.PI
                              * hullArea
                              / (
                            perimeter
                            * perimeter
                    );

            RotatedRect minimumAreaRectangle =
                    Imgproc.minAreaRect(
                            hullContour2f
                    );

            double shorterSide =
                    Math.min(
                            minimumAreaRectangle
                                    .size.width,
                            minimumAreaRectangle
                                    .size.height
                    );

            double longerSide =
                    Math.max(
                            minimumAreaRectangle
                                    .size.width,
                            minimumAreaRectangle
                                    .size.height
                    );

            double orientedAspectRatio =
                    shorterSide <= 0.0
                            ? Double.POSITIVE_INFINITY
                            : longerSide
                              / shorterSide;

            boolean nearRoundTablet =
                    hullPoints.size()
                            >= ROUND_TABLET_MIN_HULL_POINTS
                            && orientedAspectRatio
                            <= ROUND_TABLET_MAX_ASPECT_RATIO
                            && circularity
                            >= ROUND_TABLET_MIN_CIRCULARITY;

            boolean circleApplied =
                    false;

            boolean ellipseApplied =
                    false;

            boolean ovalApplied =
                    false;

            OvalTabletCandidate ovalCandidate =
                    null;

            boolean elongatedEllipseApplied =
                    false;

            ElongatedEllipseTabletCandidate
                    elongatedEllipseCandidate =
                    null;

            boolean triangleApplied =
                    false;

            TriangleTabletCandidate triangleCandidate =
                    null;

            boolean capsuleApplied =
                    false;

            CapsuleShapeCandidate capsuleCandidate =
                    null;

            double ellipseAreaRatio =
                    0.0;

            boolean trianglePriorityCandidate =
                    orientedAspectRatio <= 1.35;

            if (trianglePriorityCandidate) {
                triangleCandidate =
                        detectTriangleTabletShape(
                                sourceBitmap,
                                pixelBox,
                                hullContour,
                                hullArea,
                                width,
                                height
                        );

                if (triangleCandidate != null
                        && triangleCandidate.contourPoints != null
                        && triangleCandidate.contourPoints.length >= 3) {
                    MatOfPoint triangleContour =
                            new MatOfPoint();

                    try {
                        triangleContour.fromArray(
                                triangleCandidate.contourPoints
                        );

                        Imgproc.fillConvexPoly(
                                mask,
                                triangleContour,
                                new Scalar(255),
                                Imgproc.LINE_8,
                                0
                        );

                        triangleApplied =
                                Core.countNonZero(
                                        mask
                                ) > 0;

                    } finally {
                        triangleContour.release();
                    }
                }
            }

            RoundTabletCircleCandidate
                    circleCandidate =
                    null;

            if (!triangleApplied) {
                circleCandidate =
                        detectRoundTabletCircle(
                                sourceBitmap,
                                pixelBox,
                                hullContour,
                                hullArea,
                                width,
                                height
                        );

                if (circleCandidate != null) {
                    int finalRadius =
                            Math.max(
                                    1,
                                    (int) Math.round(
                                            circleCandidate.radius
                                                    * ROUND_TABLET_CIRCLE_RADIUS_SCALE
                                    )
                            );

                    Imgproc.circle(
                            mask,
                            circleCandidate.center,
                            finalRadius,
                            new Scalar(255),
                            -1,
                            Imgproc.LINE_8,
                            0
                    );

                    circleApplied =
                            true;
                }
            }

            if (!triangleApplied
                    && !circleApplied) {
                elongatedEllipseCandidate =
                        detectElongatedEllipseTabletShape(
                                sourceBitmap,
                                hullContour,
                                hullArea,
                                circularity,
                                orientedAspectRatio,
                                width,
                                height
                        );

                if (elongatedEllipseCandidate != null) {
                    double ellipseWidthScale =
                            elongatedEllipseCandidate.width
                                    <= elongatedEllipseCandidate.height
                                    ? ELLIPSE_TABLET_MINOR_SCALE
                                    : ELLIPSE_TABLET_MAJOR_SCALE;

                    double ellipseHeightScale =
                            elongatedEllipseCandidate.height
                                    <= elongatedEllipseCandidate.width
                                    ? ELLIPSE_TABLET_MINOR_SCALE
                                    : ELLIPSE_TABLET_MAJOR_SCALE;

                    Imgproc.ellipse(
                            mask,
                            elongatedEllipseCandidate.center,
                            new org.opencv.core.Size(
                                    elongatedEllipseCandidate.width
                                            * ellipseWidthScale
                                            * 0.5,
                                    elongatedEllipseCandidate.height
                                            * ellipseHeightScale
                                            * 0.5
                            ),
                            elongatedEllipseCandidate.angle,
                            0.0,
                            360.0,
                            new Scalar(255),
                            -1,
                            Imgproc.LINE_8,
                            0
                    );

                    elongatedEllipseApplied =
                            Core.countNonZero(
                                    mask
                            ) > 0;
                }
            }

            if (!trianglePriorityCandidate
                    && !circleApplied
                    && !elongatedEllipseApplied) {
                triangleCandidate =
                        detectTriangleTabletShape(
                                sourceBitmap,
                                pixelBox,
                                hullContour,
                                hullArea,
                                width,
                                height
                        );

                if (triangleCandidate != null
                        && triangleCandidate.contourPoints != null
                        && triangleCandidate.contourPoints.length >= 3) {
                    MatOfPoint triangleContour =
                            new MatOfPoint();

                    try {
                        triangleContour.fromArray(
                                triangleCandidate.contourPoints
                        );

                        Imgproc.fillConvexPoly(
                                mask,
                                triangleContour,
                                new Scalar(255),
                                Imgproc.LINE_8,
                                0
                        );

                        triangleApplied =
                                Core.countNonZero(
                                        mask
                                ) > 0;

                    } finally {
                        triangleContour.release();
                    }
                }
            }

            if (!circleApplied
                    && !elongatedEllipseApplied
                    && !triangleApplied) {
                ovalCandidate =
                        detectOvalTabletShape(
                                sourceBitmap,
                                pixelBox,
                                hullContour,
                                hullArea,
                                orientedAspectRatio,
                                width,
                                height
                        );

                if (ovalCandidate != null) {
                    double ovalWidthScale =
                            ovalCandidate.width
                                    >= ovalCandidate.height
                                    ? OVAL_TABLET_MAJOR_SCALE
                                    : OVAL_TABLET_MINOR_SCALE;

                    double ovalHeightScale =
                            ovalCandidate.height
                                    >= ovalCandidate.width
                                    ? OVAL_TABLET_MAJOR_SCALE
                                    : OVAL_TABLET_MINOR_SCALE;

                    Imgproc.ellipse(
                            mask,
                            ovalCandidate.center,
                            new org.opencv.core.Size(
                                    ovalCandidate.width
                                            * ovalWidthScale
                                            * 0.5,
                                    ovalCandidate.height
                                            * ovalHeightScale
                                            * 0.5
                            ),
                            ovalCandidate.angle,
                            0.0,
                            360.0,
                            new Scalar(255),
                            -1,
                            Imgproc.LINE_8,
                            0
                    );

                    ovalApplied =
                            Core.countNonZero(
                                    mask
                            ) > 0;
                }
            }

            if (!circleApplied
                    && !elongatedEllipseApplied
                    && !triangleApplied
                    && !ovalApplied
                    && nearRoundTablet
                    && hullPoints.size() >= 5) {
                try {
                    RotatedRect fittedEllipse =
                            Imgproc.fitEllipse(
                                    hullContour2f
                            );

                    double ellipseWidth =
                            fittedEllipse
                                    .size.width;

                    double ellipseHeight =
                            fittedEllipse
                                    .size.height;

                    double ellipseShorterSide =
                            Math.min(
                                    ellipseWidth,
                                    ellipseHeight
                            );

                    double ellipseLongerSide =
                            Math.max(
                                    ellipseWidth,
                                    ellipseHeight
                            );

                    double ellipseAspectRatio =
                            ellipseShorterSide <= 0.0
                                    ? Double.POSITIVE_INFINITY
                                    : ellipseLongerSide
                                      / ellipseShorterSide;

                    double ellipseArea =
                            Math.PI
                                    * ellipseWidth
                                    * 0.5
                                    * ellipseHeight
                                    * 0.5;

                    ellipseAreaRatio =
                            ellipseArea
                                    / Math.max(
                                    1.0,
                                    hullArea
                            );

                    org.opencv.core.Rect
                            hullBounds =
                            Imgproc.boundingRect(
                                    hullContour
                            );

                    double centerMarginX =
                            Math.max(
                                    2.0,
                                    hullBounds.width
                                            * 0.20
                            );

                    double centerMarginY =
                            Math.max(
                                    2.0,
                                    hullBounds.height
                                            * 0.20
                            );

                    boolean centerValid =
                            fittedEllipse.center.x
                                    >= hullBounds.x
                                    - centerMarginX
                                    && fittedEllipse.center.x
                                    <= hullBounds.x
                                    + hullBounds.width
                                    + centerMarginX
                                    && fittedEllipse.center.y
                                    >= hullBounds.y
                                    - centerMarginY
                                    && fittedEllipse.center.y
                                    <= hullBounds.y
                                    + hullBounds.height
                                    + centerMarginY;

                    boolean ellipseValid =
                            ellipseWidth > 1.0
                                    && ellipseHeight > 1.0
                                    && ellipseAspectRatio
                                    <= ROUND_TABLET_MAX_ASPECT_RATIO
                                    && ellipseAreaRatio
                                    >= ROUND_TABLET_MIN_ELLIPSE_AREA_RATIO
                                    && ellipseAreaRatio
                                    <= ROUND_TABLET_MAX_ELLIPSE_AREA_RATIO
                                    && centerValid;

                    if (ellipseValid) {
                        Imgproc.ellipse(
                                mask,
                                fittedEllipse.center,
                                new org.opencv.core.Size(
                                        ellipseWidth
                                                * 0.5,
                                        ellipseHeight
                                                * 0.5
                                ),
                                fittedEllipse.angle,
                                0.0,
                                360.0,
                                new Scalar(255),
                                -1,
                                Imgproc.LINE_8,
                                0
                        );

                        ellipseApplied =
                                true;
                    }

                } catch (Exception ellipseError) {
                    Log.w(
                            "PillMask",
                            "Az ellipszis-illesztés nem "
                                    + "használható, marad a "
                                    + "konvex burok.",
                            ellipseError
                    );
                }
            }

            if (!circleApplied
                    && !elongatedEllipseApplied
                    && !triangleApplied
                    && !ovalApplied
                    && !ellipseApplied
                    && orientedAspectRatio
                    >= CAPSULE_MIN_GEMINI_HULL_ASPECT_RATIO) {
                capsuleCandidate =
                        detectCapsuleShape(
                                sourceBitmap,
                                pixelBox,
                                hullContour,
                                hullArea,
                                orientedAspectRatio,
                                width,
                                height
                        );

                if (capsuleCandidate != null) {
                    drawCapsuleMask(
                            mask,
                            capsuleCandidate.center,
                            capsuleCandidate.majorLength
                                    * CAPSULE_LENGTH_SCALE,
                            capsuleCandidate.minorWidth
                                    * CAPSULE_WIDTH_SCALE,
                            capsuleCandidate.majorAxisAngleDegrees
                    );

                    capsuleApplied =
                            Core.countNonZero(
                                    mask
                            ) > 0;
                }
            }

            if (!circleApplied
                    && !elongatedEllipseApplied
                    && !triangleApplied
                    && !ovalApplied
                    && !ellipseApplied
                    && !capsuleApplied) {
                Imgproc.fillPoly(
                        mask,
                        Collections.singletonList(
                                hullContour
                        ),
                        new Scalar(255)
                );
            }

            if (Core.countNonZero(mask) <= 0) {
                throw new Exception(
                        "A létrehozott gyógyszermaszk üres."
                );
            }

            String maskMode;

            if (circleApplied) {
                maskMode =
                        "OPENCV_CIRCLE";

            } else if (elongatedEllipseApplied) {
                maskMode =
                        "OPENCV_ELLIPSE_TABLET";

            } else if (triangleApplied) {
                maskMode =
                        "OPENCV_TRIANGLE_TABLET";

            } else if (ovalApplied) {
                maskMode =
                        "OPENCV_OVAL_TABLET";

            } else if (ellipseApplied) {
                maskMode =
                        "ROUND_ELLIPSE";

            } else if (capsuleApplied) {
                maskMode =
                        "OPENCV_CAPSULE";

            } else {
                maskMode =
                        "CONVEX_HULL";
            }

            String circleLog =
                    circleCandidate == null
                            ? "; OpenCV-kör=nincs"
                            : "; OpenCV-kör=("
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            circleCandidate.center.x
                    )
                              + ","
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            circleCandidate.center.y
                    )
                              + "), r="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            circleCandidate.radius
                    )
                              + ", él="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            circleCandidate.edgeSupport
                    )
                              + ", hull-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            circleCandidate.hullCoverage
                    )
                              + ", kör-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            circleCandidate.circleCoverage
                    );

            String elongatedEllipseLog =
                    elongatedEllipseCandidate == null
                            ? "; OpenCV-ellipszis=nincs"
                            : "; OpenCV-ellipszis=("
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            elongatedEllipseCandidate.center.x
                    )
                              + ","
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            elongatedEllipseCandidate.center.y
                    )
                              + "), méret="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            elongatedEllipseCandidate.width
                    )
                              + "x"
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            elongatedEllipseCandidate.height
                    )
                              + ", oldalarány="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            elongatedEllipseCandidate.aspectRatio
                    )
                              + ", ellipszis/hull="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            elongatedEllipseCandidate.ellipseAreaRatio
                    )
                              + ", fényeltolás="
                              + String.format(
                            Locale.ROOT,
                            "%.2f",
                            elongatedEllipseCandidate.centerShift
                    )
                              + ", hull-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            elongatedEllipseCandidate.hullCoverage
                    )
                              + ", jelölt-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            elongatedEllipseCandidate.candidateCoverage
                    );

            String triangleLog =
                    triangleCandidate == null
                            ? "; OpenCV-háromszög=nincs"
                            : "; OpenCV-háromszög=("
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            triangleCandidate.center.x
                    )
                              + ","
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            triangleCandidate.center.y
                    )
                              + "), terület="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            triangleCandidate.hullArea
                    )
                              + ", szilárdság="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            triangleCandidate.solidity
                    )
                              + ", háromszög/hull="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            triangleCandidate.triangleToHullAreaRatio
                    )
                              + ", oldalarány="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            triangleCandidate.orientedAspectRatio
                    )
                              + ", Gemini-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            triangleCandidate.geminiHullCoverage
                    )
                              + ", jelölt-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            triangleCandidate.candidateCoverage
                    );

            String ovalLog =
                    ovalCandidate == null
                            ? "; OpenCV-ovális=nincs"
                            : "; OpenCV-ovális=("
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ovalCandidate.center.x
                    )
                              + ","
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ovalCandidate.center.y
                    )
                              + "), méret="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ovalCandidate.width
                    )
                              + "x"
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ovalCandidate.height
                    )
                              + ", oldalarány="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            ovalCandidate.aspectRatio
                    )
                              + ", hull-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            ovalCandidate.hullCoverage
                    )
                              + ", ovális-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            ovalCandidate.ellipseCoverage
                    )
                              + ", él-támogatás="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            ovalCandidate.edgeSupport
                    );

            String capsuleLog =
                    capsuleCandidate == null
                            ? "; OpenCV-kapszula=nincs"
                            : "; OpenCV-kapszula=("
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            capsuleCandidate.center.x
                    )
                              + ","
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            capsuleCandidate.center.y
                    )
                              + "), hossz="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            capsuleCandidate.majorLength
                    )
                              + ", szélesség="
                              + String.format(
                            Locale.ROOT,
                            "%.1f",
                            capsuleCandidate.minorWidth
                    )
                              + ", oldalarány="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            capsuleCandidate.aspectRatio
                    )
                              + ", hull-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            capsuleCandidate.hullCoverage
                    )
                              + ", kapszula-lefedés="
                              + String.format(
                            Locale.ROOT,
                            "%.3f",
                            capsuleCandidate.candidateCoverage
                    );

            Log.d(
                    "PillMask",
                    "Maszk elkészült. mód="
                            + maskMode
                            + "; Gemini pontok="
                            + safePoints.size()
                            + "; hull pontok="
                            + hullPoints.size()
                            + "; eredeti terület="
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            originalArea
                    )
                            + "; hull terület="
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            hullArea
                    )
                            + "; köralakúság="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            circularity
                    )
                            + "; orientált oldalarány="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            orientedAspectRatio
                    )
                            + "; ellipszis/hull="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            ellipseAreaRatio
                    )
                            + circleLog
                            + elongatedEllipseLog
                            + triangleLog
                            + ovalLog
                            + capsuleLog
            );

            return mask;

        } catch (Exception error) {
            mask.release();
            throw error;

        } finally {
            originalContour.release();
            hullIndices.release();
            hullContour.release();
            hullContour2f.release();
        }
    }

    private RoundTabletCircleCandidate detectRoundTabletCircle(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint hullContour,
            double hullArea,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || pixelBox == null
                || hullContour == null
                || hullContour.empty()
                || hullArea <= 0.0) {
            return null;
        }

        int boxLeft =
                Math.max(
                        0,
                        Math.min(
                                imageWidth - 1,
                                pixelBox.left
                        )
                );

        int boxTop =
                Math.max(
                        0,
                        Math.min(
                                imageHeight - 1,
                                pixelBox.top
                        )
                );

        int boxRight =
                Math.max(
                        boxLeft + 1,
                        Math.min(
                                imageWidth,
                                pixelBox.right
                        )
                );

        int boxBottom =
                Math.max(
                        boxTop + 1,
                        Math.min(
                                imageHeight,
                                pixelBox.bottom
                        )
                );

        int boxWidth =
                boxRight - boxLeft;

        int boxHeight =
                boxBottom - boxTop;

        int shorterBoxSide =
                Math.min(
                        boxWidth,
                        boxHeight
                );

        int longerBoxSide =
                Math.max(
                        boxWidth,
                        boxHeight
                );

        if (shorterBoxSide < 12) {
            return null;
        }

        double boxAspectRatio =
                longerBoxSide
                        / (double) shorterBoxSide;

        if (boxAspectRatio
                > ROUND_TABLET_CIRCLE_MAX_BOX_ASPECT_RATIO) {
            return null;
        }

        int padding =
                Math.max(
                        4,
                        (int) Math.round(
                                longerBoxSide
                                        * 0.25
                        )
                );

        int roiLeft =
                Math.max(
                        0,
                        boxLeft - padding
                );

        int roiTop =
                Math.max(
                        0,
                        boxTop - padding
                );

        int roiRight =
                Math.min(
                        imageWidth,
                        boxRight + padding
                );

        int roiBottom =
                Math.min(
                        imageHeight,
                        boxBottom + padding
                );

        int roiWidth =
                roiRight - roiLeft;

        int roiHeight =
                roiBottom - roiTop;

        if (roiWidth < 16
                || roiHeight < 16) {
            return null;
        }

        int minimumRadius =
                Math.max(
                        4,
                        (int) Math.round(
                                shorterBoxSide
                                        * ROUND_TABLET_CIRCLE_MIN_RADIUS_RATIO
                        )
                );

        int maximumRadius =
                Math.max(
                        minimumRadius + 1,
                        (int) Math.round(
                                shorterBoxSide
                                        * ROUND_TABLET_CIRCLE_MAX_RADIUS_RATIO
                        )
                );

        Mat sourceMat =
                new Mat();

        Mat grayMat =
                new Mat();

        Mat grayRoi =
                new Mat();

        Mat filteredRoi =
                new Mat();

        Mat edgeRoi =
                new Mat();

        Mat circles =
                new Mat();

        Mat hullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat circleMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        try {
            Utils.bitmapToMat(
                    sourceBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        grayMat,
                        Imgproc.COLOR_RGBA2GRAY
                );

            } else if (sourceMat.channels() == 3) {
                Imgproc.cvtColor(
                        sourceMat,
                        grayMat,
                        Imgproc.COLOR_RGB2GRAY
                );

            } else {
                sourceMat.copyTo(
                        grayMat
                );
            }

            org.opencv.core.Rect roiRect =
                    new org.opencv.core.Rect(
                            roiLeft,
                            roiTop,
                            roiWidth,
                            roiHeight
                    );

            grayRoi =
                    grayMat.submat(
                            roiRect
                    );

            Imgproc.medianBlur(
                    grayRoi,
                    filteredRoi,
                    5
            );

            Imgproc.Canny(
                    filteredRoi,
                    edgeRoi,
                    40.0,
                    120.0
            );

            Imgproc.HoughCircles(
                    filteredRoi,
                    circles,
                    Imgproc.HOUGH_GRADIENT,
                    1.0,
                    Math.max(
                            8.0,
                            shorterBoxSide * 0.50
                    ),
                    80.0,
                    16.0,
                    minimumRadius,
                    maximumRadius
            );

            if (circles.empty()
                    || circles.cols() <= 0) {
                return null;
            }

            Imgproc.fillPoly(
                    hullMask,
                    Collections.singletonList(
                            hullContour
                    ),
                    new Scalar(255)
            );

            double boxCenterX =
                    (
                            boxLeft
                                    + boxRight
                    ) * 0.5;

            double boxCenterY =
                    (
                            boxTop
                                    + boxBottom
                    ) * 0.5;

            double maximumCenterOffset =
                    longerBoxSide
                            * ROUND_TABLET_CIRCLE_MAX_CENTER_OFFSET_RATIO;

            double boxArea =
                    Math.max(
                            1.0,
                            (double) boxWidth
                                    * boxHeight
                    );

            RoundTabletCircleCandidate
                    bestCandidate =
                    null;

            for (int index = 0;
                 index < circles.cols();
                 index++) {
                double[] values =
                        circles.get(
                                0,
                                index
                        );

                if (values == null
                        || values.length < 3) {
                    continue;
                }

                double localCenterX =
                        values[0];

                double localCenterY =
                        values[1];

                double radius =
                        values[2];

                if (radius
                        < minimumRadius
                        || radius
                        > maximumRadius) {
                    continue;
                }

                double globalCenterX =
                        roiLeft
                                + localCenterX;

                double globalCenterY =
                        roiTop
                                + localCenterY;

                double centerDx =
                        globalCenterX
                                - boxCenterX;

                double centerDy =
                        globalCenterY
                                - boxCenterY;

                double centerOffset =
                        Math.sqrt(
                                centerDx * centerDx
                                        + centerDy * centerDy
                        );

                if (centerOffset
                        > maximumCenterOffset) {
                    continue;
                }

                double circleArea =
                        Math.PI
                                * radius
                                * radius;

                double circleToBoxRatio =
                        circleArea
                                / boxArea;

                if (circleToBoxRatio < 0.45
                        || circleToBoxRatio > 1.05) {
                    continue;
                }

                circleMask.setTo(
                        new Scalar(0)
                );

                Imgproc.circle(
                        circleMask,
                        new Point(
                                globalCenterX,
                                globalCenterY
                        ),
                        Math.max(
                                1,
                                (int) Math.round(
                                        radius
                                )
                        ),
                        new Scalar(255),
                        -1,
                        Imgproc.LINE_8,
                        0
                );

                Core.bitwise_and(
                        hullMask,
                        circleMask,
                        intersectionMask
                );

                double intersectionArea =
                        Core.countNonZero(
                                intersectionMask
                        );

                double circlePixelArea =
                        Core.countNonZero(
                                circleMask
                        );

                double hullCoverage =
                        intersectionArea
                                / Math.max(
                                1.0,
                                hullArea
                        );

                double circleCoverage =
                        intersectionArea
                                / Math.max(
                                1.0,
                                circlePixelArea
                        );

                if (hullCoverage
                        < ROUND_TABLET_CIRCLE_MIN_HULL_COVERAGE
                        || circleCoverage
                        < ROUND_TABLET_CIRCLE_MIN_CIRCLE_COVERAGE) {
                    continue;
                }

                double edgeSupport =
                        calculateCircleEdgeSupport(
                                edgeRoi,
                                localCenterX,
                                localCenterY,
                                radius
                        );

                if (edgeSupport
                        < ROUND_TABLET_CIRCLE_MIN_EDGE_SUPPORT) {
                    continue;
                }

                double normalizedCenterOffset =
                        centerOffset
                                / Math.max(
                                1.0,
                                maximumCenterOffset
                        );

                double score =
                        edgeSupport * 4.0
                                + hullCoverage * 2.0
                                + circleCoverage * 2.0
                                - normalizedCenterOffset;

                RoundTabletCircleCandidate
                        candidate =
                        new RoundTabletCircleCandidate(
                                new Point(
                                        globalCenterX,
                                        globalCenterY
                                ),
                                radius,
                                edgeSupport,
                                hullCoverage,
                                circleCoverage,
                                score
                        );

                if (bestCandidate == null
                        || candidate.score
                        > bestCandidate.score) {
                    bestCandidate =
                            candidate;
                }
            }

            return bestCandidate;

        } catch (Exception circleError) {
            Log.w(
                    "PillMask",
                    "Az OpenCV-kördetektálás nem "
                            + "használható, marad a Gemini-"
                            + "alapú maszk.",
                    circleError
            );

            return null;

        } finally {
            sourceMat.release();
            grayMat.release();

            if (grayRoi != null) {
                grayRoi.release();
            }

            filteredRoi.release();
            edgeRoi.release();
            circles.release();
            hullMask.release();
            circleMask.release();
            intersectionMask.release();
        }
    }

    private double calculateCircleEdgeSupport(
            Mat edgeImage,
            double centerX,
            double centerY,
            double radius
    ) {
        if (edgeImage == null
                || edgeImage.empty()
                || radius <= 1.0) {
            return 0.0;
        }

        int supportedSamples =
                0;

        for (int sampleIndex = 0;
             sampleIndex < ROUND_TABLET_CIRCLE_SAMPLE_COUNT;
             sampleIndex++) {
            double angle =
                    2.0
                            * Math.PI
                            * sampleIndex
                            / ROUND_TABLET_CIRCLE_SAMPLE_COUNT;

            double cosine =
                    Math.cos(
                            angle
                    );

            double sine =
                    Math.sin(
                            angle
                    );

            boolean edgeFound =
                    false;

            for (int radialOffset = -3;
                 radialOffset <= 3;
                 radialOffset++) {
                double testedRadius =
                        radius
                                + radialOffset;

                int x =
                        (int) Math.round(
                                centerX
                                        + testedRadius
                                        * cosine
                        );

                int y =
                        (int) Math.round(
                                centerY
                                        + testedRadius
                                        * sine
                        );

                if (x < 0
                        || x >= edgeImage.cols()
                        || y < 0
                        || y >= edgeImage.rows()) {
                    continue;
                }

                double[] edgeValue =
                        edgeImage.get(
                                y,
                                x
                        );

                if (edgeValue != null
                        && edgeValue.length > 0
                        && edgeValue[0] > 0.0) {
                    edgeFound =
                            true;

                    break;
                }
            }

            if (edgeFound) {
                supportedSamples++;
            }
        }

        return supportedSamples
                / (double) ROUND_TABLET_CIRCLE_SAMPLE_COUNT;
    }

    private static final class RoundTabletCircleCandidate {
        final Point center;
        final double radius;
        final double edgeSupport;
        final double hullCoverage;
        final double circleCoverage;
        final double score;

        RoundTabletCircleCandidate(
                Point center,
                double radius,
                double edgeSupport,
                double hullCoverage,
                double circleCoverage,
                double score
        ) {
            this.center =
                    center;

            this.radius =
                    radius;

            this.edgeSupport =
                    edgeSupport;

            this.hullCoverage =
                    hullCoverage;

            this.circleCoverage =
                    circleCoverage;

            this.score =
                    score;
        }
    }

    private ElongatedEllipseTabletCandidate
    detectElongatedEllipseTabletShape(
            Bitmap sourceBitmap,
            MatOfPoint hullContour,
            double hullArea,
            double circularity,
            double orientedAspectRatio,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || hullContour == null
                || hullContour.empty()
                || hullArea <= 0.0
                || orientedAspectRatio
                < ELLIPSE_TABLET_MIN_ORIENTED_ASPECT_RATIO
                || orientedAspectRatio
                > ELLIPSE_TABLET_MAX_ORIENTED_ASPECT_RATIO
                || circularity
                < ELLIPSE_TABLET_MIN_CIRCULARITY
                || circularity
                > ELLIPSE_TABLET_MAX_CIRCULARITY) {
            return null;
        }

        Point[] hullPoints =
                hullContour.toArray();

        if (hullPoints.length < 5) {
            return null;
        }

        org.opencv.core.Rect hullBounds =
                Imgproc.boundingRect(
                        hullContour
                );

        if (hullBounds.width < 8
                || hullBounds.height < 8) {
            return null;
        }

        double boundingFillRatio =
                hullArea
                        / Math.max(
                        1.0,
                        (double) hullBounds.width
                                * hullBounds.height
                );

        if (boundingFillRatio
                < ELLIPSE_TABLET_MIN_BOUNDING_FILL_RATIO
                || boundingFillRatio
                > ELLIPSE_TABLET_MAX_BOUNDING_FILL_RATIO) {
            return null;
        }

        MatOfPoint2f hullContour2f =
                new MatOfPoint2f(
                        hullPoints
                );

        Mat sourceMat =
                new Mat();

        Mat rgbMat =
                new Mat();

        Mat labMat =
                new Mat();

        Mat lightnessMat =
                new Mat();

        Mat lightnessRoi =
                new Mat();

        Mat localHullMask =
                new Mat();

        Mat fullHullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat candidateMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        try {
            RotatedRect fittedEllipse =
                    Imgproc.fitEllipse(
                            hullContour2f
                    );

            double ellipseWidth =
                    fittedEllipse.size.width;

            double ellipseHeight =
                    fittedEllipse.size.height;

            double ellipseShorterSide =
                    Math.min(
                            ellipseWidth,
                            ellipseHeight
                    );

            double ellipseLongerSide =
                    Math.max(
                            ellipseWidth,
                            ellipseHeight
                    );

            if (ellipseShorterSide <= 1.0
                    || ellipseLongerSide <= 1.0) {
                return null;
            }

            double fittedAspectRatio =
                    ellipseLongerSide
                            / ellipseShorterSide;

            if (fittedAspectRatio
                    < ELLIPSE_TABLET_MIN_FITTED_ASPECT_RATIO
                    || fittedAspectRatio
                    > ELLIPSE_TABLET_MAX_FITTED_ASPECT_RATIO) {
                return null;
            }

            double fittedEllipseArea =
                    Math.PI
                            * ellipseWidth
                            * 0.5
                            * ellipseHeight
                            * 0.5;

            double ellipseAreaRatio =
                    fittedEllipseArea
                            / Math.max(
                            1.0,
                            hullArea
                    );

            if (ellipseAreaRatio
                    < ELLIPSE_TABLET_MIN_ELLIPSE_AREA_RATIO
                    || ellipseAreaRatio
                    > ELLIPSE_TABLET_MAX_ELLIPSE_AREA_RATIO) {
                return null;
            }

            int longerHullSide =
                    Math.max(
                            hullBounds.width,
                            hullBounds.height
                    );

            int padding =
                    Math.max(
                            5,
                            (int) Math.round(
                                    longerHullSide
                                            * ELLIPSE_TABLET_ROI_PADDING_RATIO
                            )
                    );

            int roiLeft =
                    Math.max(
                            0,
                            hullBounds.x - padding
                    );

            int roiTop =
                    Math.max(
                            0,
                            hullBounds.y - padding
                    );

            int roiRight =
                    Math.min(
                            imageWidth,
                            hullBounds.x
                                    + hullBounds.width
                                    + padding
                    );

            int roiBottom =
                    Math.min(
                            imageHeight,
                            hullBounds.y
                                    + hullBounds.height
                                    + padding
                    );

            int roiWidth =
                    roiRight - roiLeft;

            int roiHeight =
                    roiBottom - roiTop;

            if (roiWidth < 12
                    || roiHeight < 12) {
                return null;
            }

            Utils.bitmapToMat(
                    sourceBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_RGBA2RGB
                );

            } else if (sourceMat.channels() == 3) {
                sourceMat.copyTo(
                        rgbMat
                );

            } else {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_GRAY2RGB
                );
            }

            Imgproc.cvtColor(
                    rgbMat,
                    labMat,
                    Imgproc.COLOR_RGB2Lab
            );

            Core.extractChannel(
                    labMat,
                    lightnessMat,
                    0
            );

            lightnessRoi =
                    lightnessMat.submat(
                            new org.opencv.core.Rect(
                                    roiLeft,
                                    roiTop,
                                    roiWidth,
                                    roiHeight
                            )
                    );

            Point[] localHullPoints =
                    new Point[
                            hullPoints.length
                            ];

            for (int pointIndex = 0;
                 pointIndex < hullPoints.length;
                 pointIndex++) {
                localHullPoints[pointIndex] =
                        new Point(
                                hullPoints[pointIndex].x
                                        - roiLeft,
                                hullPoints[pointIndex].y
                                        - roiTop
                        );
            }

            MatOfPoint localHullContour =
                    new MatOfPoint(
                            localHullPoints
                    );

            try {
                localHullMask =
                        Mat.zeros(
                                roiHeight,
                                roiWidth,
                                CvType.CV_8UC1
                        );

                Imgproc.fillConvexPoly(
                        localHullMask,
                        localHullContour,
                        new Scalar(255),
                        Imgproc.LINE_8,
                        0
                );

            } finally {
                localHullContour.release();
            }

            int borderThickness =
                    Math.max(
                            2,
                            (int) Math.round(
                                    Math.min(
                                            roiWidth,
                                            roiHeight
                                    )
                                            * ELLIPSE_TABLET_BACKGROUND_BORDER_RATIO
                            )
                    );

            byte[] lightnessPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                            ];

            byte[] hullMaskPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                            ];

            lightnessRoi.get(
                    0,
                    0,
                    lightnessPixels
            );

            localHullMask.get(
                    0,
                    0,
                    hullMaskPixels
            );

            int maximumBorderPixelCount =
                    roiWidth
                            * roiHeight;

            int[] borderLightnessValues =
                    new int[
                            maximumBorderPixelCount
                            ];

            int borderPixelCount =
                    0;

            for (int localY = 0;
                 localY < roiHeight;
                 localY++) {
                for (int localX = 0;
                     localX < roiWidth;
                     localX++) {
                    boolean borderPixel =
                            localX < borderThickness
                                    || localY < borderThickness
                                    || localX >= roiWidth
                                    - borderThickness
                                    || localY >= roiHeight
                                    - borderThickness;

                    if (!borderPixel) {
                        continue;
                    }

                    borderLightnessValues[
                            borderPixelCount
                            ] =
                            lightnessPixels[
                                    localY
                                            * roiWidth
                                            + localX
                                    ] & 0xFF;

                    borderPixelCount++;
                }
            }

            if (borderPixelCount < 8) {
                return null;
            }

            borderLightnessValues =
                    java.util.Arrays.copyOf(
                            borderLightnessValues,
                            borderPixelCount
                    );

            java.util.Arrays.sort(
                    borderLightnessValues
            );

            double backgroundLightness =
                    medianOfSortedIntArray(
                            borderLightnessValues
                    );

            double weightedXSum =
                    0.0;

            double weightedYSum =
                    0.0;

            double brightnessWeightSum =
                    0.0;

            int hullPixelCount =
                    0;

            for (int localY = 0;
                 localY < roiHeight;
                 localY++) {
                for (int localX = 0;
                     localX < roiWidth;
                     localX++) {
                    int pixelIndex =
                            localY
                                    * roiWidth
                                    + localX;

                    if ((
                            hullMaskPixels[pixelIndex]
                                    & 0xFF
                    ) == 0) {
                        continue;
                    }

                    hullPixelCount++;

                    double lightness =
                            lightnessPixels[pixelIndex]
                                    & 0xFF;

                    double brightnessWeight =
                            Math.max(
                                    0.0,
                                    lightness
                                            - backgroundLightness
                            );

                    brightnessWeight =
                            Math.min(
                                    ELLIPSE_TABLET_MAX_PIXEL_BRIGHTNESS_WEIGHT,
                                    brightnessWeight
                            );

                    if (brightnessWeight <= 0.0) {
                        continue;
                    }

                    double globalX =
                            roiLeft
                                    + localX;

                    double globalY =
                            roiTop
                                    + localY;

                    weightedXSum +=
                            globalX
                                    * brightnessWeight;

                    weightedYSum +=
                            globalY
                                    * brightnessWeight;

                    brightnessWeightSum +=
                            brightnessWeight;
                }
            }

            if (hullPixelCount <= 0
                    || brightnessWeightSum
                    / hullPixelCount
                    < ELLIPSE_TABLET_MIN_AVERAGE_BRIGHTNESS_WEIGHT) {
                return null;
            }

            double brightCenterX =
                    weightedXSum
                            / brightnessWeightSum;

            double brightCenterY =
                    weightedYSum
                            / brightnessWeightSum;

            double minorAxisAngleDegrees =
                    ellipseWidth
                            <= ellipseHeight
                            ? fittedEllipse.angle
                            : fittedEllipse.angle
                              + 90.0;

            double minorAxisAngleRadians =
                    Math.toRadians(
                            minorAxisAngleDegrees
                    );

            double minorDirectionX =
                    Math.cos(
                            minorAxisAngleRadians
                    );

            double minorDirectionY =
                    Math.sin(
                            minorAxisAngleRadians
                    );

            double centerDifferenceX =
                    brightCenterX
                            - fittedEllipse.center.x;

            double centerDifferenceY =
                    brightCenterY
                            - fittedEllipse.center.y;

            double projectedCenterDifference =
                    centerDifferenceX
                            * minorDirectionX
                            + centerDifferenceY
                            * minorDirectionY;

            double maximumCenterShift =
                    ellipseShorterSide
                            * ELLIPSE_TABLET_MAX_CENTER_SHIFT_MINOR_RATIO;

            double limitedCenterShift =
                    Math.max(
                            -maximumCenterShift,
                            Math.min(
                                    maximumCenterShift,
                                    projectedCenterDifference
                            )
                    )
                            * ELLIPSE_TABLET_CENTER_SHIFT_WEIGHT;

            Point correctedCenter =
                    new Point(
                            fittedEllipse.center.x
                                    + minorDirectionX
                                    * limitedCenterShift,
                            fittedEllipse.center.y
                                    + minorDirectionY
                                    * limitedCenterShift
                    );

            double widthScale =
                    ellipseWidth
                            <= ellipseHeight
                            ? ELLIPSE_TABLET_MINOR_SCALE
                            : ELLIPSE_TABLET_MAJOR_SCALE;

            double heightScale =
                    ellipseHeight
                            <= ellipseWidth
                            ? ELLIPSE_TABLET_MINOR_SCALE
                            : ELLIPSE_TABLET_MAJOR_SCALE;

            Imgproc.fillConvexPoly(
                    fullHullMask,
                    hullContour,
                    new Scalar(255),
                    Imgproc.LINE_8,
                    0
            );

            Imgproc.ellipse(
                    candidateMask,
                    correctedCenter,
                    new org.opencv.core.Size(
                            ellipseWidth
                                    * widthScale
                                    * 0.5,
                            ellipseHeight
                                    * heightScale
                                    * 0.5
                    ),
                    fittedEllipse.angle,
                    0.0,
                    360.0,
                    new Scalar(255),
                    -1,
                    Imgproc.LINE_8,
                    0
            );

            Core.bitwise_and(
                    fullHullMask,
                    candidateMask,
                    intersectionMask
            );

            double intersectionArea =
                    Core.countNonZero(
                            intersectionMask
                    );

            double hullMaskArea =
                    Core.countNonZero(
                            fullHullMask
                    );

            double candidateArea =
                    Core.countNonZero(
                            candidateMask
                    );

            double hullCoverage =
                    intersectionArea
                            / Math.max(
                            1.0,
                            hullMaskArea
                    );

            double candidateCoverage =
                    intersectionArea
                            / Math.max(
                            1.0,
                            candidateArea
                    );

            if (hullCoverage
                    < ELLIPSE_TABLET_MIN_HULL_COVERAGE
                    || candidateCoverage
                    < ELLIPSE_TABLET_MIN_CANDIDATE_COVERAGE) {
                return null;
            }

            Log.d(
                    "PillMask",
                    "Ellipszis tabletta fényességalapú finomítása "
                            + "elfogadva. középpont=("
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            correctedCenter.x
                    )
                            + ","
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            correctedCenter.y
                    )
                            + "); méret="
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ellipseWidth
                    )
                            + "x"
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            ellipseHeight
                    )
                            + "; oldalarány="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            fittedAspectRatio
                    )
                            + "; rövidtengely-eltolás="
                            + String.format(
                            Locale.ROOT,
                            "%.2f",
                            limitedCenterShift
                    )
                            + "; hull-lefedés="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            hullCoverage
                    )
                            + "; jelölt-lefedés="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            candidateCoverage
                    )
            );

            return new ElongatedEllipseTabletCandidate(
                    correctedCenter,
                    ellipseWidth,
                    ellipseHeight,
                    fittedEllipse.angle,
                    fittedAspectRatio,
                    ellipseAreaRatio,
                    boundingFillRatio,
                    limitedCenterShift,
                    hullCoverage,
                    candidateCoverage
            );

        } catch (Exception ellipseTabletError) {
            Log.w(
                    "PillMask",
                    "Az ellipszis alakú tabletta külön finomítása "
                            + "nem használható, marad a korábbi "
                            + "alakfelismerési sorrend.",
                    ellipseTabletError
            );

            return null;

        } finally {
            hullContour2f.release();
            sourceMat.release();
            rgbMat.release();
            labMat.release();
            lightnessMat.release();

            if (lightnessRoi != null) {
                lightnessRoi.release();
            }

            if (localHullMask != null) {
                localHullMask.release();
            }

            fullHullMask.release();
            candidateMask.release();
            intersectionMask.release();
        }
    }

    private static final class ElongatedEllipseTabletCandidate {
        final Point center;
        final double width;
        final double height;
        final double angle;
        final double aspectRatio;
        final double ellipseAreaRatio;
        final double boundingFillRatio;
        final double centerShift;
        final double hullCoverage;
        final double candidateCoverage;

        ElongatedEllipseTabletCandidate(
                Point center,
                double width,
                double height,
                double angle,
                double aspectRatio,
                double ellipseAreaRatio,
                double boundingFillRatio,
                double centerShift,
                double hullCoverage,
                double candidateCoverage
        ) {
            this.center =
                    center;

            this.width =
                    width;

            this.height =
                    height;

            this.angle =
                    angle;

            this.aspectRatio =
                    aspectRatio;

            this.ellipseAreaRatio =
                    ellipseAreaRatio;

            this.boundingFillRatio =
                    boundingFillRatio;

            this.centerShift =
                    centerShift;

            this.hullCoverage =
                    hullCoverage;

            this.candidateCoverage =
                    candidateCoverage;
        }
    }

    private TriangleTabletCandidate detectTriangleTabletShape(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint geminiHullContour,
            double geminiHullArea,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || pixelBox == null
                || geminiHullContour == null
                || geminiHullContour.empty()
                || geminiHullArea <= 0.0) {
            return null;
        }

        int boxLeft =
                Math.max(
                        0,
                        Math.min(
                                imageWidth - 1,
                                pixelBox.left
                        )
                );

        int boxTop =
                Math.max(
                        0,
                        Math.min(
                                imageHeight - 1,
                                pixelBox.top
                        )
                );

        int boxRight =
                Math.max(
                        boxLeft + 1,
                        Math.min(
                                imageWidth,
                                pixelBox.right
                        )
                );

        int boxBottom =
                Math.max(
                        boxTop + 1,
                        Math.min(
                                imageHeight,
                                pixelBox.bottom
                        )
                );

        int boxWidth =
                boxRight - boxLeft;

        int boxHeight =
                boxBottom - boxTop;

        int shorterBoxSide =
                Math.min(
                        boxWidth,
                        boxHeight
                );

        int longerBoxSide =
                Math.max(
                        boxWidth,
                        boxHeight
                );

        if (shorterBoxSide < 12) {
            return null;
        }

        double boxAspectRatio =
                longerBoxSide
                        / (double) shorterBoxSide;

        if (boxAspectRatio
                > TRIANGLE_TABLET_MAX_BOX_ASPECT_RATIO) {
            return null;
        }

        int padding =
                Math.max(
                        6,
                        (int) Math.round(
                                longerBoxSide
                                        * TRIANGLE_TABLET_ROI_PADDING_RATIO
                        )
                );

        int roiLeft =
                Math.max(
                        0,
                        boxLeft - padding
                );

        int roiTop =
                Math.max(
                        0,
                        boxTop - padding
                );

        int roiRight =
                Math.min(
                        imageWidth,
                        boxRight + padding
                );

        int roiBottom =
                Math.min(
                        imageHeight,
                        boxBottom + padding
                );

        int roiWidth =
                roiRight - roiLeft;

        int roiHeight =
                roiBottom - roiTop;

        if (roiWidth < 18
                || roiHeight < 18) {
            return null;
        }

        Mat sourceMat =
                new Mat();

        Mat rgbMat =
                new Mat();

        Mat labMat =
                new Mat();

        Mat labRoi =
                new Mat();

        Mat seedMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat borderMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat foregroundMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat hierarchy =
                new Mat();

        Mat openKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                3,
                                3
                        )
                );

        Mat closeKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                5,
                                5
                        )
                );

        Mat geminiHullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat candidateMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        List<MatOfPoint> contours =
                new ArrayList<>();

        try {
            Point localBoxCenter =
                    new Point(
                            (
                                    boxLeft
                                            + boxRight
                            ) * 0.5
                                    - roiLeft,
                            (
                                    boxTop
                                            + boxBottom
                            ) * 0.5
                                    - roiTop
                    );

            int seedRadius =
                    Math.max(
                            3,
                            (int) Math.round(
                                    shorterBoxSide
                                            * TRIANGLE_TABLET_SEED_RADIUS_RATIO
                            )
                    );

            Imgproc.circle(
                    seedMask,
                    localBoxCenter,
                    seedRadius,
                    new Scalar(255),
                    -1,
                    Imgproc.LINE_8,
                    0
            );

            int borderThickness =
                    Math.max(
                            2,
                            (int) Math.round(
                                    Math.min(
                                            roiWidth,
                                            roiHeight
                                    ) * 0.10
                            )
                    );

            Imgproc.rectangle(
                    borderMask,
                    new Point(
                            0,
                            0
                    ),
                    new Point(
                            roiWidth - 1,
                            roiHeight - 1
                    ),
                    new Scalar(255),
                    borderThickness
            );

            Utils.bitmapToMat(
                    sourceBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_RGBA2RGB
                );

            } else if (sourceMat.channels() == 3) {
                sourceMat.copyTo(
                        rgbMat
                );

            } else {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_GRAY2RGB
                );
            }

            Imgproc.cvtColor(
                    rgbMat,
                    labMat,
                    Imgproc.COLOR_RGB2Lab
            );

            labRoi =
                    labMat.submat(
                            new org.opencv.core.Rect(
                                    roiLeft,
                                    roiTop,
                                    roiWidth,
                                    roiHeight
                            )
                    );

            double[] foregroundLab =
                    calculateMaskedLabMedian(
                            labRoi,
                            seedMask
                    );

            double[] backgroundLab =
                    calculateMaskedLabMedian(
                            labRoi,
                            borderMask
                    );

            if (foregroundLab == null
                    || backgroundLab == null) {
                return null;
            }

            double foregroundLightnessDifference =
                    foregroundLab[0]
                            - backgroundLab[0];

            double foregroundADifference =
                    foregroundLab[1]
                            - backgroundLab[1];

            double foregroundBDifference =
                    foregroundLab[2]
                            - backgroundLab[2];

            double foregroundBackgroundDistance =
                    Math.sqrt(
                            TRIANGLE_TABLET_LAB_LIGHTNESS_WEIGHT
                                    * foregroundLightnessDifference
                                    * foregroundLightnessDifference
                                    + foregroundADifference
                                    * foregroundADifference
                                    + foregroundBDifference
                                    * foregroundBDifference
                    );

            if (foregroundBackgroundDistance
                    < TRIANGLE_TABLET_MIN_LAB_FOREGROUND_BACKGROUND_DISTANCE) {
                return null;
            }

            double maximumForegroundDistance =
                    Math.min(
                            TRIANGLE_TABLET_LAB_MAX_MAX_DISTANCE,
                            Math.max(
                                    TRIANGLE_TABLET_LAB_MIN_MAX_DISTANCE,
                                    foregroundBackgroundDistance
                                            * TRIANGLE_TABLET_LAB_MAX_DISTANCE_SCALE
                            )
                    );

            byte[] labPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                                    * 3
                            ];

            labRoi.get(
                    0,
                    0,
                    labPixels
            );

            byte[] foregroundPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                            ];

            for (int pixelIndex = 0;
                 pixelIndex < foregroundPixels.length;
                 pixelIndex++) {
                int labOffset =
                        pixelIndex * 3;

                double lightness =
                        labPixels[
                                labOffset
                                ] & 0xFF;

                double aChannel =
                        labPixels[
                                labOffset + 1
                                ] & 0xFF;

                double bChannel =
                        labPixels[
                                labOffset + 2
                                ] & 0xFF;

                double foregroundLightness =
                        lightness
                                - foregroundLab[0];

                double foregroundA =
                        aChannel
                                - foregroundLab[1];

                double foregroundB =
                        bChannel
                                - foregroundLab[2];

                double backgroundLightness =
                        lightness
                                - backgroundLab[0];

                double backgroundA =
                        aChannel
                                - backgroundLab[1];

                double backgroundB =
                        bChannel
                                - backgroundLab[2];

                double foregroundDistance =
                        Math.sqrt(
                                TRIANGLE_TABLET_LAB_LIGHTNESS_WEIGHT
                                        * foregroundLightness
                                        * foregroundLightness
                                        + foregroundA
                                        * foregroundA
                                        + foregroundB
                                        * foregroundB
                        );

                double backgroundDistance =
                        Math.sqrt(
                                TRIANGLE_TABLET_LAB_LIGHTNESS_WEIGHT
                                        * backgroundLightness
                                        * backgroundLightness
                                        + backgroundA
                                        * backgroundA
                                        + backgroundB
                                        * backgroundB
                        );

                double foregroundChromaDistance =
                        Math.sqrt(
                                foregroundA
                                        * foregroundA
                                        + foregroundB
                                        * foregroundB
                        );

                double backgroundChromaDistance =
                        Math.sqrt(
                                backgroundA
                                        * backgroundA
                                        + backgroundB
                                        * backgroundB
                        );

                boolean weightedColorCloser =
                        foregroundDistance
                                < backgroundDistance
                                * TRIANGLE_TABLET_LAB_FOREGROUND_TO_BACKGROUND_RATIO;

                boolean chromaCloser =
                        foregroundChromaDistance
                                < backgroundChromaDistance
                                * TRIANGLE_TABLET_CHROMA_FOREGROUND_TO_BACKGROUND_RATIO;

                boolean foreground =
                        foregroundDistance
                                <= maximumForegroundDistance
                                && (
                                weightedColorCloser
                                        || chromaCloser
                        );

                foregroundPixels[
                        pixelIndex
                        ] =
                        foreground
                                ? (byte) 0xFF
                                : 0;
            }

            foregroundMask.put(
                    0,
                    0,
                    foregroundPixels
            );

            Core.bitwise_or(
                    foregroundMask,
                    seedMask,
                    foregroundMask
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
            );

            Imgproc.findContours(
                    foregroundMask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
            );

            if (contours.isEmpty()) {
                return null;
            }

            Imgproc.fillPoly(
                    geminiHullMask,
                    Collections.singletonList(
                            geminiHullContour
                    ),
                    new Scalar(255)
            );

            double boxCenterX =
                    (
                            boxLeft
                                    + boxRight
                    ) * 0.5;

            double boxCenterY =
                    (
                            boxTop
                                    + boxBottom
                    ) * 0.5;

            double maximumCenterOffset =
                    longerBoxSide
                            * TRIANGLE_TABLET_MAX_CENTER_OFFSET_RATIO;

            double boxArea =
                    Math.max(
                            1.0,
                            (double) boxWidth
                                    * boxHeight
                    );

            TriangleTabletCandidate bestCandidate =
                    null;

            for (MatOfPoint contour
                    : contours) {
                double contourArea =
                        Math.abs(
                                Imgproc.contourArea(
                                        contour
                                )
                        );

                double contourAreaToBoxRatio =
                        contourArea
                                / boxArea;

                if (contourAreaToBoxRatio
                        < TRIANGLE_TABLET_MIN_CONTOUR_AREA_TO_BOX_RATIO
                        || contourAreaToBoxRatio
                        > TRIANGLE_TABLET_MAX_CONTOUR_AREA_TO_BOX_RATIO) {
                    continue;
                }

                MatOfInt hullIndices =
                        new MatOfInt();

                MatOfPoint localHull =
                        new MatOfPoint();

                MatOfPoint2f localHull2f =
                        new MatOfPoint2f();

                MatOfPoint2f approximatedTriangle2f =
                        new MatOfPoint2f();

                MatOfPoint globalHull =
                        new MatOfPoint();

                try {
                    Imgproc.convexHull(
                            contour,
                            hullIndices
                    );

                    int[] indices =
                            hullIndices.toArray();

                    Point[] contourPoints =
                            contour.toArray();

                    if (indices.length < 3) {
                        continue;
                    }

                    List<Point> localHullPoints =
                            new ArrayList<>(
                                    indices.length
                            );

                    for (int pointIndex
                            : indices) {
                        if (pointIndex >= 0
                                && pointIndex < contourPoints.length) {
                            localHullPoints.add(
                                    contourPoints[
                                            pointIndex
                                            ]
                            );
                        }
                    }

                    if (localHullPoints.size() < 3) {
                        continue;
                    }

                    localHull.fromList(
                            localHullPoints
                    );

                    localHull2f.fromList(
                            localHullPoints
                    );

                    double candidateHullArea =
                            Math.abs(
                                    Imgproc.contourArea(
                                            localHull
                                    )
                            );

                    if (candidateHullArea <= 0.0) {
                        continue;
                    }

                    double solidity =
                            contourArea
                                    / candidateHullArea;

                    if (solidity
                            < TRIANGLE_TABLET_MIN_SOLIDITY) {
                        continue;
                    }

                    org.opencv.core.Rect
                            candidateBounds =
                            Imgproc.boundingRect(
                                    localHull
                            );

                    double hullFillRatio =
                            candidateHullArea
                                    / Math.max(
                                    1.0,
                                    (double) candidateBounds.width
                                            * candidateBounds.height
                            );

                    if (hullFillRatio
                            < TRIANGLE_TABLET_MIN_HULL_FILL_RATIO
                            || hullFillRatio
                            > TRIANGLE_TABLET_MAX_HULL_FILL_RATIO) {
                        continue;
                    }

                    RotatedRect minimumAreaRectangle =
                            Imgproc.minAreaRect(
                                    localHull2f
                            );

                    double shorterSide =
                            Math.min(
                                    minimumAreaRectangle.size.width,
                                    minimumAreaRectangle.size.height
                            );

                    double longerSide =
                            Math.max(
                                    minimumAreaRectangle.size.width,
                                    minimumAreaRectangle.size.height
                            );

                    double orientedAspectRatio =
                            shorterSide <= 0.0
                                    ? Double.POSITIVE_INFINITY
                                    : longerSide
                                      / shorterSide;

                    if (orientedAspectRatio
                            > TRIANGLE_TABLET_MAX_ORIENTED_ASPECT_RATIO) {
                        continue;
                    }

                    double globalCenterX =
                            roiLeft
                                    + minimumAreaRectangle.center.x;

                    double globalCenterY =
                            roiTop
                                    + minimumAreaRectangle.center.y;

                    double centerDx =
                            globalCenterX
                                    - boxCenterX;

                    double centerDy =
                            globalCenterY
                                    - boxCenterY;

                    double centerOffset =
                            Math.sqrt(
                                    centerDx * centerDx
                                            + centerDy * centerDy
                            );

                    if (centerOffset
                            > maximumCenterOffset) {
                        continue;
                    }

                    double hullPerimeter =
                            Imgproc.arcLength(
                                    localHull2f,
                                    true
                            );

                    Point[] triangleVertices =
                            new Point[0];

                    double[] approximationRatios =
                            new double[]{
                                    0.040,
                                    TRIANGLE_TABLET_APPROX_EPSILON_RATIO,
                                    0.060,
                                    0.070
                            };

                    for (double approximationRatio
                            : approximationRatios) {
                        Imgproc.approxPolyDP(
                                localHull2f,
                                approximatedTriangle2f,
                                hullPerimeter
                                        * approximationRatio,
                                true
                        );

                        Point[] testedVertices =
                                approximatedTriangle2f.toArray();

                        if (testedVertices.length == 3) {
                            triangleVertices =
                                    testedVertices;

                            break;
                        }
                    }

                    if (triangleVertices.length != 3) {
                        continue;
                    }

                    if (Imgproc.pointPolygonTest(
                            localHull2f,
                            localBoxCenter,
                            false
                    ) < 0.0) {
                        continue;
                    }

                    double triangleArea =
                            Math.abs(
                                    Imgproc.contourArea(
                                            approximatedTriangle2f
                                    )
                            );

                    double triangleToHullAreaRatio =
                            triangleArea
                                    / candidateHullArea;

                    if (triangleToHullAreaRatio
                            < TRIANGLE_TABLET_MIN_TRIANGLE_TO_HULL_AREA_RATIO
                            || triangleToHullAreaRatio
                            > TRIANGLE_TABLET_MAX_TRIANGLE_TO_HULL_AREA_RATIO) {
                        continue;
                    }

                    double shortestTriangleSide =
                            Double.POSITIVE_INFINITY;

                    double longestTriangleSide =
                            0.0;

                    boolean anglesValid =
                            true;

                    for (int vertexIndex = 0;
                         vertexIndex < 3;
                         vertexIndex++) {
                        Point previousVertex =
                                triangleVertices[
                                        (
                                                vertexIndex + 2
                                        ) % 3
                                        ];

                        Point currentVertex =
                                triangleVertices[
                                        vertexIndex
                                        ];

                        Point nextVertex =
                                triangleVertices[
                                        (
                                                vertexIndex + 1
                                        ) % 3
                                        ];

                        double firstVectorX =
                                previousVertex.x
                                        - currentVertex.x;

                        double firstVectorY =
                                previousVertex.y
                                        - currentVertex.y;

                        double secondVectorX =
                                nextVertex.x
                                        - currentVertex.x;

                        double secondVectorY =
                                nextVertex.y
                                        - currentVertex.y;

                        double firstLength =
                                Math.sqrt(
                                        firstVectorX
                                                * firstVectorX
                                                + firstVectorY
                                                * firstVectorY
                                );

                        double secondLength =
                                Math.sqrt(
                                        secondVectorX
                                                * secondVectorX
                                                + secondVectorY
                                                * secondVectorY
                                );

                        if (firstLength <= 0.0
                                || secondLength <= 0.0) {
                            anglesValid =
                                    false;

                            break;
                        }

                        double cosine =
                                (
                                        firstVectorX
                                                * secondVectorX
                                                + firstVectorY
                                                * secondVectorY
                                )
                                        / (
                                        firstLength
                                                * secondLength
                                );

                        cosine =
                                Math.max(
                                        -1.0,
                                        Math.min(
                                                1.0,
                                                cosine
                                        )
                                );

                        double angleDegrees =
                                Math.toDegrees(
                                        Math.acos(
                                                cosine
                                        )
                                );

                        if (angleDegrees
                                < TRIANGLE_TABLET_MIN_VERTEX_ANGLE_DEGREES
                                || angleDegrees
                                > TRIANGLE_TABLET_MAX_VERTEX_ANGLE_DEGREES) {
                            anglesValid =
                                    false;

                            break;
                        }

                        double sideLength =
                                pointDistance(
                                        currentVertex,
                                        nextVertex
                                );

                        shortestTriangleSide =
                                Math.min(
                                        shortestTriangleSide,
                                        sideLength
                                );

                        longestTriangleSide =
                                Math.max(
                                        longestTriangleSide,
                                        sideLength
                                );
                    }

                    if (!anglesValid
                            || shortestTriangleSide <= 0.0) {
                        continue;
                    }

                    double triangleSideRatio =
                            longestTriangleSide
                                    / shortestTriangleSide;

                    if (triangleSideRatio
                            > TRIANGLE_TABLET_MAX_SIDE_RATIO) {
                        continue;
                    }

                    Point[] globalHullPoints =
                            new Point[
                                    localHullPoints.size()
                                    ];

                    for (int pointIndex = 0;
                         pointIndex < localHullPoints.size();
                         pointIndex++) {
                        Point localPoint =
                                localHullPoints.get(
                                        pointIndex
                                );

                        globalHullPoints[
                                pointIndex
                                ] =
                                new Point(
                                        roiLeft
                                                + localPoint.x,
                                        roiTop
                                                + localPoint.y
                                );
                    }

                    globalHull.fromArray(
                            globalHullPoints
                    );

                    candidateMask.setTo(
                            new Scalar(0)
                    );

                    Imgproc.fillConvexPoly(
                            candidateMask,
                            globalHull,
                            new Scalar(255),
                            Imgproc.LINE_8,
                            0
                    );

                    Core.bitwise_and(
                            geminiHullMask,
                            candidateMask,
                            intersectionMask
                    );

                    double intersectionArea =
                            Core.countNonZero(
                                    intersectionMask
                            );

                    double candidateArea =
                            Core.countNonZero(
                                    candidateMask
                            );

                    double geminiHullCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    geminiHullArea
                            );

                    double candidateCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    candidateArea
                            );

                    if (geminiHullCoverage
                            < TRIANGLE_TABLET_MIN_GEMINI_HULL_COVERAGE
                            || candidateCoverage
                            < TRIANGLE_TABLET_MIN_CANDIDATE_COVERAGE) {
                        continue;
                    }

                    double normalizedCenterOffset =
                            centerOffset
                                    / Math.max(
                                    1.0,
                                    maximumCenterOffset
                            );

                    double score =
                            geminiHullCoverage * 3.0
                                    + candidateCoverage * 2.0
                                    + solidity
                                    + triangleToHullAreaRatio
                                    + (
                                    1.0
                                            - Math.min(
                                            1.0,
                                            triangleSideRatio
                                                    / TRIANGLE_TABLET_MAX_SIDE_RATIO
                                    )
                            )
                                    - normalizedCenterOffset;

                    TriangleTabletCandidate candidate =
                            new TriangleTabletCandidate(
                                    globalHullPoints,
                                    new Point(
                                            globalCenterX,
                                            globalCenterY
                                    ),
                                    candidateHullArea,
                                    solidity,
                                    triangleToHullAreaRatio,
                                    orientedAspectRatio,
                                    triangleSideRatio,
                                    foregroundBackgroundDistance,
                                    geminiHullCoverage,
                                    candidateCoverage,
                                    score
                            );

                    if (bestCandidate == null
                            || candidate.score
                            > bestCandidate.score) {
                        bestCandidate =
                                candidate;
                    }

                } finally {
                    hullIndices.release();
                    localHull.release();
                    localHull2f.release();
                    approximatedTriangle2f.release();
                    globalHull.release();
                }
            }

            if (bestCandidate != null) {
                Log.d(
                        "PillMask",
                        "Háromszög tabletta Lab-finomítás elfogadva. "
                                + "középpont=("
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.x
                        )
                                + ","
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.y
                        )
                                + "); hull terület="
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.hullArea
                        )
                                + "; háromszög/hull="
                                + String.format(
                                Locale.ROOT,
                                "%.3f",
                                bestCandidate.triangleToHullAreaRatio
                        )
                                + "; Gemini-lefedés="
                                + String.format(
                                Locale.ROOT,
                                "%.3f",
                                bestCandidate.geminiHullCoverage
                        )
                                + "; Lab FG/BG="
                                + String.format(
                                Locale.ROOT,
                                "%.2f",
                                bestCandidate.foregroundBackgroundDistance
                        )
                );
            }

            return bestCandidate;

        } catch (Exception triangleError) {
            Log.w(
                    "PillMask",
                    "A háromszög tabletta Lab-finomítása "
                            + "nem használható, marad a korábbi "
                            + "maszklogika.",
                    triangleError
            );

            return null;

        } finally {
            sourceMat.release();
            rgbMat.release();
            labMat.release();

            if (labRoi != null) {
                labRoi.release();
            }

            seedMask.release();
            borderMask.release();
            foregroundMask.release();
            hierarchy.release();
            openKernel.release();
            closeKernel.release();
            geminiHullMask.release();
            candidateMask.release();
            intersectionMask.release();

            for (MatOfPoint contour
                    : contours) {
                contour.release();
            }
        }
    }

    private static final class TriangleTabletCandidate {
        final Point[] contourPoints;
        final Point center;
        final double hullArea;
        final double solidity;
        final double triangleToHullAreaRatio;
        final double orientedAspectRatio;
        final double triangleSideRatio;
        final double foregroundBackgroundDistance;
        final double geminiHullCoverage;
        final double candidateCoverage;
        final double score;

        TriangleTabletCandidate(
                Point[] contourPoints,
                Point center,
                double hullArea,
                double solidity,
                double triangleToHullAreaRatio,
                double orientedAspectRatio,
                double triangleSideRatio,
                double foregroundBackgroundDistance,
                double geminiHullCoverage,
                double candidateCoverage,
                double score
        ) {
            this.contourPoints =
                    contourPoints == null
                            ? new Point[0]
                            : contourPoints.clone();

            this.center =
                    center;

            this.hullArea =
                    hullArea;

            this.solidity =
                    solidity;

            this.triangleToHullAreaRatio =
                    triangleToHullAreaRatio;

            this.orientedAspectRatio =
                    orientedAspectRatio;

            this.triangleSideRatio =
                    triangleSideRatio;

            this.foregroundBackgroundDistance =
                    foregroundBackgroundDistance;

            this.geminiHullCoverage =
                    geminiHullCoverage;

            this.candidateCoverage =
                    candidateCoverage;

            this.score =
                    score;
        }
    }

    private OvalTabletCandidate detectOvalTabletShape(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint hullContour,
            double hullArea,
            double geminiHullAspectRatio,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || pixelBox == null
                || hullContour == null
                || hullContour.empty()
                || hullArea <= 0.0) {
            return null;
        }

        int boxLeft =
                Math.max(
                        0,
                        Math.min(
                                imageWidth - 1,
                                pixelBox.left
                        )
                );

        int boxTop =
                Math.max(
                        0,
                        Math.min(
                                imageHeight - 1,
                                pixelBox.top
                        )
                );

        int boxRight =
                Math.max(
                        boxLeft + 1,
                        Math.min(
                                imageWidth,
                                pixelBox.right
                        )
                );

        int boxBottom =
                Math.max(
                        boxTop + 1,
                        Math.min(
                                imageHeight,
                                pixelBox.bottom
                        )
                );

        int boxWidth =
                boxRight - boxLeft;

        int boxHeight =
                boxBottom - boxTop;

        int shorterBoxSide =
                Math.min(
                        boxWidth,
                        boxHeight
                );

        int longerBoxSide =
                Math.max(
                        boxWidth,
                        boxHeight
                );

        if (shorterBoxSide < 10) {
            return null;
        }

        double boxAspectRatio =
                longerBoxSide
                        / (double) shorterBoxSide;

        if (boxAspectRatio
                > OVAL_TABLET_MAX_BOX_ASPECT_RATIO) {
            return null;
        }

        org.opencv.core.Rect hullBounds =
                Imgproc.boundingRect(
                        hullContour
                );

        int objectLeft =
                Math.min(
                        boxLeft,
                        hullBounds.x
                );

        int objectTop =
                Math.min(
                        boxTop,
                        hullBounds.y
                );

        int objectRight =
                Math.max(
                        boxRight,
                        hullBounds.x
                                + hullBounds.width
                );

        int objectBottom =
                Math.max(
                        boxBottom,
                        hullBounds.y
                                + hullBounds.height
                );

        int padding =
                Math.max(
                        6,
                        (int) Math.round(
                                longerBoxSide
                                        * OVAL_TABLET_ROI_PADDING_RATIO
                        )
                );

        int roiLeft =
                Math.max(
                        0,
                        objectLeft - padding
                );

        int roiTop =
                Math.max(
                        0,
                        objectTop - padding
                );

        int roiRight =
                Math.min(
                        imageWidth,
                        objectRight + padding
                );

        int roiBottom =
                Math.min(
                        imageHeight,
                        objectBottom + padding
                );

        int roiWidth =
                roiRight - roiLeft;

        int roiHeight =
                roiBottom - roiTop;

        if (roiWidth < 18
                || roiHeight < 18) {
            return null;
        }

        Mat sourceMat =
                new Mat();

        Mat grayMat =
                new Mat();

        Mat grayRoi =
                new Mat();

        Mat blurredRoi =
                new Mat();

        Mat lowThresholdEdges =
                new Mat();

        Mat highThresholdEdges =
                new Mat();

        Mat combinedEdges =
                new Mat();

        Mat contourInput =
                new Mat();

        Mat hierarchy =
                new Mat();

        Mat fullHullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat candidateMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        List<MatOfPoint> contours =
                new ArrayList<>();

        try {
            Utils.bitmapToMat(
                    sourceBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        grayMat,
                        Imgproc.COLOR_RGBA2GRAY
                );

            } else if (sourceMat.channels() == 3) {
                Imgproc.cvtColor(
                        sourceMat,
                        grayMat,
                        Imgproc.COLOR_RGB2GRAY
                );

            } else {
                sourceMat.copyTo(
                        grayMat
                );
            }

            grayRoi =
                    grayMat.submat(
                            new org.opencv.core.Rect(
                                    roiLeft,
                                    roiTop,
                                    roiWidth,
                                    roiHeight
                            )
                    );

            Imgproc.GaussianBlur(
                    grayRoi,
                    blurredRoi,
                    new org.opencv.core.Size(
                            5,
                            5
                    ),
                    0.0
            );

            Imgproc.Canny(
                    blurredRoi,
                    lowThresholdEdges,
                    15.0,
                    45.0
            );

            Imgproc.Canny(
                    blurredRoi,
                    highThresholdEdges,
                    25.0,
                    75.0
            );

            Core.bitwise_or(
                    lowThresholdEdges,
                    highThresholdEdges,
                    combinedEdges
            );

            contourInput =
                    combinedEdges.clone();

            Imgproc.findContours(
                    contourInput,
                    contours,
                    hierarchy,
                    Imgproc.RETR_LIST,
                    Imgproc.CHAIN_APPROX_NONE
            );

            if (contours.isEmpty()) {
                return null;
            }

            Imgproc.fillPoly(
                    fullHullMask,
                    Collections.singletonList(
                            hullContour
                    ),
                    new Scalar(255)
            );

            double boxCenterX =
                    (
                            boxLeft
                                    + boxRight
                    ) * 0.5;

            double boxCenterY =
                    (
                            boxTop
                                    + boxBottom
                    ) * 0.5;

            double localBoxCenterX =
                    boxCenterX
                            - roiLeft;

            double localBoxCenterY =
                    boxCenterY
                            - roiTop;

            double maximumCenterOffset =
                    longerBoxSide
                            * OVAL_TABLET_MAX_CENTER_OFFSET_RATIO;

            double boxArea =
                    Math.max(
                            1.0,
                            (double) boxWidth
                                    * boxHeight
                    );

            OvalTabletCandidate bestCandidate =
                    null;

            for (MatOfPoint contour
                    : contours) {
                Point[] contourPoints =
                        contour.toArray();

                if (contourPoints.length < 24) {
                    continue;
                }

                MatOfPoint2f contour2f =
                        new MatOfPoint2f(
                                contourPoints
                        );

                try {
                    RotatedRect localEllipse =
                            Imgproc.fitEllipse(
                                    contour2f
                            );

                    double ellipseWidth =
                            localEllipse.size.width;

                    double ellipseHeight =
                            localEllipse.size.height;

                    double shorterSide =
                            Math.min(
                                    ellipseWidth,
                                    ellipseHeight
                            );

                    double longerSide =
                            Math.max(
                                    ellipseWidth,
                                    ellipseHeight
                            );

                    if (shorterSide <= 2.0
                            || longerSide <= 2.0) {
                        continue;
                    }

                    double aspectRatio =
                            longerSide
                                    / shorterSide;

                    if (aspectRatio
                            < OVAL_TABLET_MIN_FITTED_ASPECT_RATIO
                            || aspectRatio
                            > OVAL_TABLET_MAX_FITTED_ASPECT_RATIO) {
                        continue;
                    }

                    double centerDx =
                            localEllipse.center.x
                                    - localBoxCenterX;

                    double centerDy =
                            localEllipse.center.y
                                    - localBoxCenterY;

                    double centerOffset =
                            Math.sqrt(
                                    centerDx * centerDx
                                            + centerDy * centerDy
                            );

                    if (centerOffset
                            > maximumCenterOffset) {
                        continue;
                    }

                    double majorToBoxRatio =
                            longerSide
                                    / Math.max(
                                    1.0,
                                    longerBoxSide
                            );

                    double minorToBoxRatio =
                            shorterSide
                                    / Math.max(
                                    1.0,
                                    shorterBoxSide
                            );

                    if (majorToBoxRatio
                            < OVAL_TABLET_MIN_MAJOR_TO_BOX_RATIO
                            || majorToBoxRatio
                            > OVAL_TABLET_MAX_MAJOR_TO_BOX_RATIO
                            || minorToBoxRatio
                            < OVAL_TABLET_MIN_MINOR_TO_BOX_RATIO
                            || minorToBoxRatio
                            > OVAL_TABLET_MAX_MINOR_TO_BOX_RATIO) {
                        continue;
                    }

                    double ellipseArea =
                            Math.PI
                                    * ellipseWidth
                                    * 0.5
                                    * ellipseHeight
                                    * 0.5;

                    double boxAreaRatio =
                            ellipseArea
                                    / boxArea;

                    if (boxAreaRatio
                            < OVAL_TABLET_MIN_BOX_AREA_RATIO
                            || boxAreaRatio
                            > OVAL_TABLET_MAX_BOX_AREA_RATIO) {
                        continue;
                    }

                    double edgeSupport =
                            calculateEllipseEdgeSupport(
                                    combinedEdges,
                                    localEllipse.center,
                                    ellipseWidth,
                                    ellipseHeight,
                                    localEllipse.angle
                            );

                    if (edgeSupport
                            < OVAL_TABLET_MIN_EDGE_SUPPORT) {
                        continue;
                    }

                    double globalCenterX =
                            roiLeft
                                    + localEllipse.center.x;

                    double globalCenterY =
                            roiTop
                                    + localEllipse.center.y;

                    double correctedCenterX =
                            globalCenterX
                                    * (
                                    1.0
                                            - OVAL_TABLET_CENTER_BOX_WEIGHT
                            )
                                    + boxCenterX
                                    * OVAL_TABLET_CENTER_BOX_WEIGHT;

                    double correctedCenterY =
                            globalCenterY
                                    * (
                                    1.0
                                            - OVAL_TABLET_CENTER_BOX_WEIGHT
                            )
                                    + boxCenterY
                                    * OVAL_TABLET_CENTER_BOX_WEIGHT;

                    candidateMask.setTo(
                            new Scalar(0)
                    );

                    Imgproc.ellipse(
                            candidateMask,
                            new Point(
                                    correctedCenterX,
                                    correctedCenterY
                            ),
                            new org.opencv.core.Size(
                                    ellipseWidth * 0.5,
                                    ellipseHeight * 0.5
                            ),
                            localEllipse.angle,
                            0.0,
                            360.0,
                            new Scalar(255),
                            -1,
                            Imgproc.LINE_8,
                            0
                    );

                    Core.bitwise_and(
                            fullHullMask,
                            candidateMask,
                            intersectionMask
                    );

                    double intersectionArea =
                            Core.countNonZero(
                                    intersectionMask
                            );

                    double candidateArea =
                            Core.countNonZero(
                                    candidateMask
                            );

                    double hullCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    hullArea
                            );

                    double ellipseCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    candidateArea
                            );

                    if (hullCoverage
                            < OVAL_TABLET_MIN_HULL_COVERAGE) {
                        continue;
                    }

                    double normalizedCenterOffset =
                            centerOffset
                                    / Math.max(
                                    1.0,
                                    maximumCenterOffset
                            );

                    double sizePenalty =
                            Math.abs(
                                    majorToBoxRatio - 0.85
                            )
                                    + Math.abs(
                                    minorToBoxRatio - 0.75
                            );

                    double score =
                            edgeSupport * 5.0
                                    + hullCoverage * 2.0
                                    + boxAreaRatio
                                    - normalizedCenterOffset
                                    - sizePenalty;

                    OvalTabletCandidate candidate =
                            new OvalTabletCandidate(
                                    new Point(
                                            correctedCenterX,
                                            correctedCenterY
                                    ),
                                    ellipseWidth,
                                    ellipseHeight,
                                    localEllipse.angle,
                                    aspectRatio,
                                    edgeSupport,
                                    hullCoverage,
                                    ellipseCoverage,
                                    boxAreaRatio,
                                    score
                            );

                    if (bestCandidate == null
                            || candidate.score
                            > bestCandidate.score) {
                        bestCandidate =
                                candidate;
                    }

                } catch (Exception fitError) {
                    Log.d(
                            "PillMask",
                            "Egy ovális élkontúr nem illeszthető.",
                            fitError
                    );

                } finally {
                    contour2f.release();
                }
            }

            if (bestCandidate != null) {
                Log.d(
                        "PillMask",
                        "Ovális tabletta élalapú finomítás elfogadva. "
                                + "középpont=("
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.x
                        )
                                + ","
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.y
                        )
                                + "); méret="
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.width
                        )
                                + "x"
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.height
                        )
                                + "; oldalarány="
                                + String.format(
                                Locale.ROOT,
                                "%.3f",
                                bestCandidate.aspectRatio
                        )
                                + "; él-támogatás="
                                + String.format(
                                Locale.ROOT,
                                "%.3f",
                                bestCandidate.edgeSupport
                        )
                                + "; Gemini-hull oldalarány="
                                + String.format(
                                Locale.ROOT,
                                "%.3f",
                                geminiHullAspectRatio
                        )
                );
            }

            return bestCandidate;

        } catch (Exception ovalError) {
            Log.w(
                    "PillMask",
                    "Az ovális tabletta élalapú finomítása "
                            + "nem használható, marad a korábbi "
                            + "maszklogika.",
                    ovalError
            );

            return null;

        } finally {
            sourceMat.release();
            grayMat.release();

            if (grayRoi != null) {
                grayRoi.release();
            }

            blurredRoi.release();
            lowThresholdEdges.release();
            highThresholdEdges.release();
            combinedEdges.release();
            contourInput.release();
            hierarchy.release();
            fullHullMask.release();
            candidateMask.release();
            intersectionMask.release();

            for (MatOfPoint contour
                    : contours) {
                contour.release();
            }
        }
    }

    private double calculateEllipseEdgeSupport(
            Mat edgeImage,
            Point center,
            double ellipseWidth,
            double ellipseHeight,
            double angleDegrees
    ) {
        if (edgeImage == null
                || edgeImage.empty()
                || center == null
                || ellipseWidth <= 2.0
                || ellipseHeight <= 2.0) {
            return 0.0;
        }

        double horizontalRadius =
                ellipseWidth * 0.5;

        double verticalRadius =
                ellipseHeight * 0.5;

        double angleRadians =
                Math.toRadians(
                        angleDegrees
                );

        double cosineAngle =
                Math.cos(
                        angleRadians
                );

        double sineAngle =
                Math.sin(
                        angleRadians
                );

        int supportedSamples =
                0;

        for (int sampleIndex = 0;
             sampleIndex
                     < OVAL_TABLET_EDGE_SAMPLE_COUNT;
             sampleIndex++) {
            double perimeterAngle =
                    2.0
                            * Math.PI
                            * sampleIndex
                            / OVAL_TABLET_EDGE_SAMPLE_COUNT;

            double cosinePerimeter =
                    Math.cos(
                            perimeterAngle
                    );

            double sinePerimeter =
                    Math.sin(
                            perimeterAngle
                    );

            boolean edgeFound =
                    false;

            for (int radialOffset = -3;
                 radialOffset <= 3;
                 radialOffset++) {
                double testedHorizontalRadius =
                        Math.max(
                                1.0,
                                horizontalRadius
                                        + radialOffset
                        );

                double testedVerticalRadius =
                        Math.max(
                                1.0,
                                verticalRadius
                                        + radialOffset
                        );

                double unrotatedX =
                        testedHorizontalRadius
                                * cosinePerimeter;

                double unrotatedY =
                        testedVerticalRadius
                                * sinePerimeter;

                int x =
                        (int) Math.round(
                                center.x
                                        + unrotatedX
                                        * cosineAngle
                                        - unrotatedY
                                        * sineAngle
                        );

                int y =
                        (int) Math.round(
                                center.y
                                        + unrotatedX
                                        * sineAngle
                                        + unrotatedY
                                        * cosineAngle
                        );

                if (x < 0
                        || x >= edgeImage.cols()
                        || y < 0
                        || y >= edgeImage.rows()) {
                    continue;
                }

                double[] edgeValue =
                        edgeImage.get(
                                y,
                                x
                        );

                if (edgeValue != null
                        && edgeValue.length > 0
                        && edgeValue[0] > 0.0) {
                    edgeFound =
                            true;

                    break;
                }
            }

            if (edgeFound) {
                supportedSamples++;
            }
        }

        return supportedSamples
                / (double) OVAL_TABLET_EDGE_SAMPLE_COUNT;
    }

    private static final class OvalTabletCandidate {
        final Point center;
        final double width;
        final double height;
        final double angle;
        final double aspectRatio;
        final double edgeSupport;
        final double hullCoverage;
        final double ellipseCoverage;
        final double boxAreaRatio;
        final double score;

        OvalTabletCandidate(
                Point center,
                double width,
                double height,
                double angle,
                double aspectRatio,
                double edgeSupport,
                double hullCoverage,
                double ellipseCoverage,
                double boxAreaRatio,
                double score
        ) {
            this.center =
                    center;

            this.width =
                    width;

            this.height =
                    height;

            this.angle =
                    angle;

            this.aspectRatio =
                    aspectRatio;

            this.edgeSupport =
                    edgeSupport;

            this.hullCoverage =
                    hullCoverage;

            this.ellipseCoverage =
                    ellipseCoverage;

            this.boxAreaRatio =
                    boxAreaRatio;

            this.score =
                    score;
        }
    }

    private CapsuleShapeCandidate detectCapsuleShape(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint hullContour,
            double hullArea,
            double geminiHullAspectRatio,
            int imageWidth,
            int imageHeight
    ) {
        CapsuleShapeCandidate colorCandidate =
                detectCapsuleShapeByLabColor(
                        sourceBitmap,
                        pixelBox,
                        hullContour,
                        hullArea,
                        geminiHullAspectRatio,
                        imageWidth,
                        imageHeight
                );

        if (colorCandidate != null) {
            return colorCandidate;
        }

        return detectCapsuleShapeByBackgroundDistance(
                sourceBitmap,
                pixelBox,
                hullContour,
                hullArea,
                geminiHullAspectRatio,
                imageWidth,
                imageHeight
        );
    }

    private CapsuleShapeCandidate detectCapsuleShapeByLabColor(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint hullContour,
            double hullArea,
            double geminiHullAspectRatio,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || pixelBox == null
                || hullContour == null
                || hullContour.empty()
                || hullArea <= 0.0
                || geminiHullAspectRatio
                < CAPSULE_MIN_GEMINI_HULL_ASPECT_RATIO) {
            return null;
        }

        int boxLeft =
                Math.max(
                        0,
                        Math.min(
                                imageWidth - 1,
                                pixelBox.left
                        )
                );

        int boxTop =
                Math.max(
                        0,
                        Math.min(
                                imageHeight - 1,
                                pixelBox.top
                        )
                );

        int boxRight =
                Math.max(
                        boxLeft + 1,
                        Math.min(
                                imageWidth,
                                pixelBox.right
                        )
                );

        int boxBottom =
                Math.max(
                        boxTop + 1,
                        Math.min(
                                imageHeight,
                                pixelBox.bottom
                        )
                );

        int boxWidth =
                boxRight - boxLeft;

        int boxHeight =
                boxBottom - boxTop;

        int longerBoxSide =
                Math.max(
                        boxWidth,
                        boxHeight
                );

        if (longerBoxSide < 12) {
            return null;
        }

        int padding =
                Math.max(
                        5,
                        (int) Math.round(
                                longerBoxSide
                                        * CAPSULE_ROI_PADDING_RATIO
                        )
                );

        int roiLeft =
                Math.max(
                        0,
                        boxLeft - padding
                );

        int roiTop =
                Math.max(
                        0,
                        boxTop - padding
                );

        int roiRight =
                Math.min(
                        imageWidth,
                        boxRight + padding
                );

        int roiBottom =
                Math.min(
                        imageHeight,
                        boxBottom + padding
                );

        int roiWidth =
                roiRight - roiLeft;

        int roiHeight =
                roiBottom - roiTop;

        if (roiWidth < 16
                || roiHeight < 16) {
            return null;
        }

        Mat sourceMat =
                new Mat();

        Mat rgbMat =
                new Mat();

        Mat labMat =
                new Mat();

        Mat labRoi =
                new Mat();

        Mat seedMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat borderMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat foregroundMask =
                Mat.zeros(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat hierarchy =
                new Mat();

        Mat openKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                3,
                                3
                        )
                );

        Mat closeKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                5,
                                5
                        )
                );

        Mat hullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat candidateMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        MatOfPoint2f hullContour2f =
                new MatOfPoint2f(
                        hullContour.toArray()
                );

        List<MatOfPoint> contours =
                new ArrayList<>();

        try {
            RotatedRect initialRectangle =
                    Imgproc.minAreaRect(
                            hullContour2f
                    );

            double initialRectangleWidth =
                    initialRectangle.size.width;

            double initialRectangleHeight =
                    initialRectangle.size.height;

            double initialMinorWidth =
                    Math.min(
                            initialRectangleWidth,
                            initialRectangleHeight
                    );

            double initialMajorLength =
                    Math.max(
                            initialRectangleWidth,
                            initialRectangleHeight
                    );

            if (initialMinorWidth <= 1.0
                    || initialMajorLength <= 1.0) {
                return null;
            }

            double initialAspectRatio =
                    initialMajorLength
                            / initialMinorWidth;

            if (initialAspectRatio
                    < CAPSULE_MIN_GEMINI_HULL_ASPECT_RATIO) {
                return null;
            }

            double initialMajorAxisAngle =
                    initialRectangleWidth
                            >= initialRectangleHeight
                            ? initialRectangle.angle
                            : initialRectangle.angle
                              + 90.0;

            Point localInitialCenter =
                    new Point(
                            initialRectangle.center.x
                                    - roiLeft,
                            initialRectangle.center.y
                                    - roiTop
                    );

            drawCapsuleMask(
                    seedMask,
                    localInitialCenter,
                    Math.max(
                            8.0,
                            initialMajorLength
                                    * CAPSULE_COLOR_SEED_LENGTH_SCALE
                    ),
                    Math.max(
                            4.0,
                            initialMinorWidth
                                    * CAPSULE_COLOR_SEED_WIDTH_SCALE
                    ),
                    initialMajorAxisAngle
            );

            int borderThickness =
                    Math.max(
                            2,
                            (int) Math.round(
                                    Math.min(
                                            roiWidth,
                                            roiHeight
                                    ) * 0.10
                            )
                    );

            Imgproc.rectangle(
                    borderMask,
                    new Point(
                            0,
                            0
                    ),
                    new Point(
                            roiWidth - 1,
                            roiHeight - 1
                    ),
                    new Scalar(255),
                    borderThickness
            );

            Utils.bitmapToMat(
                    sourceBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_RGBA2RGB
                );

            } else if (sourceMat.channels() == 3) {
                sourceMat.copyTo(
                        rgbMat
                );

            } else {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_GRAY2RGB
                );
            }

            Imgproc.cvtColor(
                    rgbMat,
                    labMat,
                    Imgproc.COLOR_RGB2Lab
            );

            labRoi =
                    labMat.submat(
                            new org.opencv.core.Rect(
                                    roiLeft,
                                    roiTop,
                                    roiWidth,
                                    roiHeight
                            )
                    );

            double[] foregroundLab =
                    calculateMaskedLabMedian(
                            labRoi,
                            seedMask
                    );

            double[] backgroundLab =
                    calculateMaskedLabMedian(
                            labRoi,
                            borderMask
                    );

            if (foregroundLab == null
                    || backgroundLab == null) {
                return null;
            }

            double foregroundBackgroundDistance =
                    labDistance(
                            foregroundLab,
                            backgroundLab
                    );

            if (foregroundBackgroundDistance
                    < CAPSULE_MIN_LAB_FOREGROUND_BACKGROUND_DISTANCE) {
                Log.d(
                        "PillMask",
                        "A kapszula és a háttér Lab-színe túl közeli; "
                                + "a korábbi kapszuladetektor marad. "
                                + "Lab-távolság="
                                + String.format(
                                Locale.ROOT,
                                "%.2f",
                                foregroundBackgroundDistance
                        )
                );

                return null;
            }

            double maximumForegroundDistance =
                    Math.min(
                            CAPSULE_LAB_MAX_MAX_DISTANCE,
                            Math.max(
                                    CAPSULE_LAB_MIN_MAX_DISTANCE,
                                    foregroundBackgroundDistance
                                            * CAPSULE_LAB_MAX_DISTANCE_SCALE
                            )
                    );

            byte[] labPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                                    * 3
                            ];

            labRoi.get(
                    0,
                    0,
                    labPixels
            );

            byte[] foregroundPixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                            ];

            for (int pixelIndex = 0;
                 pixelIndex
                         < foregroundPixels.length;
                 pixelIndex++) {
                int labOffset =
                        pixelIndex * 3;

                double lightness =
                        labPixels[
                                labOffset
                                ] & 0xFF;

                double aChannel =
                        labPixels[
                                labOffset + 1
                                ] & 0xFF;

                double bChannel =
                        labPixels[
                                labOffset + 2
                                ] & 0xFF;

                double foregroundDistance =
                        labDistance(
                                lightness,
                                aChannel,
                                bChannel,
                                foregroundLab
                        );

                double backgroundDistance =
                        labDistance(
                                lightness,
                                aChannel,
                                bChannel,
                                backgroundLab
                        );

                boolean foreground =
                        foregroundDistance
                                <= maximumForegroundDistance
                                && foregroundDistance
                                < backgroundDistance
                                * CAPSULE_LAB_FOREGROUND_TO_BACKGROUND_RATIO;

                foregroundPixels[pixelIndex] =
                        foreground
                                ? (byte) 0xFF
                                : 0;
            }

            foregroundMask.put(
                    0,
                    0,
                    foregroundPixels
            );

            Core.bitwise_or(
                    foregroundMask,
                    seedMask,
                    foregroundMask
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
            );

            Imgproc.findContours(
                    foregroundMask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
            );

            if (contours.isEmpty()) {
                return null;
            }

            Imgproc.fillPoly(
                    hullMask,
                    Collections.singletonList(
                            hullContour
                    ),
                    new Scalar(255)
            );

            double boxCenterX =
                    (
                            boxLeft
                                    + boxRight
                    ) * 0.5;

            double boxCenterY =
                    (
                            boxTop
                                    + boxBottom
                    ) * 0.5;

            double maximumCenterOffset =
                    longerBoxSide
                            * CAPSULE_MAX_CENTER_OFFSET_RATIO;

            double minimumContourArea =
                    Math.max(
                            20.0,
                            boxWidth
                                    * boxHeight
                                    * 0.08
                    );

            CapsuleShapeCandidate bestCandidate =
                    null;

            for (MatOfPoint contour : contours) {
                double contourArea =
                        Math.abs(
                                Imgproc.contourArea(
                                        contour
                                )
                        );

                if (contourArea
                        < minimumContourArea) {
                    continue;
                }

                MatOfPoint2f contour2f =
                        new MatOfPoint2f(
                                contour.toArray()
                        );

                try {
                    RotatedRect rectangle =
                            Imgproc.minAreaRect(
                                    contour2f
                            );

                    double rectangleWidth =
                            rectangle.size.width;

                    double rectangleHeight =
                            rectangle.size.height;

                    double minorWidth =
                            Math.min(
                                    rectangleWidth,
                                    rectangleHeight
                            );

                    double majorLength =
                            Math.max(
                                    rectangleWidth,
                                    rectangleHeight
                            );

                    if (minorWidth <= 1.0
                            || majorLength <= 1.0) {
                        continue;
                    }

                    double aspectRatio =
                            majorLength
                                    / minorWidth;

                    if (aspectRatio
                            < CAPSULE_MIN_IMAGE_ASPECT_RATIO
                            || aspectRatio
                            > CAPSULE_MAX_IMAGE_ASPECT_RATIO) {
                        continue;
                    }

                    double globalCenterX =
                            roiLeft
                                    + rectangle.center.x;

                    double globalCenterY =
                            roiTop
                                    + rectangle.center.y;

                    double centerDx =
                            globalCenterX
                                    - boxCenterX;

                    double centerDy =
                            globalCenterY
                                    - boxCenterY;

                    double centerOffset =
                            Math.sqrt(
                                    centerDx * centerDx
                                            + centerDy * centerDy
                            );

                    if (centerOffset
                            > maximumCenterOffset) {
                        continue;
                    }

                    double idealCapsuleArea =
                            (
                                    majorLength
                                            - minorWidth
                            ) * minorWidth
                                    + Math.PI
                                    * Math.pow(
                                    minorWidth
                                            * 0.5,
                                    2.0
                            );

                    double contourFillRatio =
                            contourArea
                                    / Math.max(
                                    1.0,
                                    idealCapsuleArea
                            );

                    if (contourFillRatio
                            < CAPSULE_MIN_CONTOUR_FILL_RATIO
                            || contourFillRatio
                            > CAPSULE_MAX_CONTOUR_FILL_RATIO) {
                        continue;
                    }

                    double majorAxisAngleDegrees =
                            rectangleWidth
                                    >= rectangleHeight
                                    ? rectangle.angle
                                    : rectangle.angle
                                      + 90.0;

                    Point globalCenter =
                            new Point(
                                    globalCenterX,
                                    globalCenterY
                            );

                    candidateMask.setTo(
                            new Scalar(0)
                    );

                    drawCapsuleMask(
                            candidateMask,
                            globalCenter,
                            majorLength,
                            minorWidth,
                            majorAxisAngleDegrees
                    );

                    Core.bitwise_and(
                            hullMask,
                            candidateMask,
                            intersectionMask
                    );

                    double intersectionArea =
                            Core.countNonZero(
                                    intersectionMask
                            );

                    double candidateArea =
                            Core.countNonZero(
                                    candidateMask
                            );

                    double hullCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    hullArea
                            );

                    double candidateCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    candidateArea
                            );

                    if (hullCoverage
                            < CAPSULE_MIN_GEMINI_HULL_COVERAGE
                            || candidateCoverage
                            < CAPSULE_MIN_CANDIDATE_COVERAGE) {
                        continue;
                    }

                    double normalizedCenterOffset =
                            centerOffset
                                    / Math.max(
                                    1.0,
                                    maximumCenterOffset
                            );

                    double score =
                            hullCoverage * 3.0
                                    + candidateCoverage * 3.0
                                    + contourFillRatio * 2.0
                                    - normalizedCenterOffset;

                    CapsuleShapeCandidate candidate =
                            new CapsuleShapeCandidate(
                                    globalCenter,
                                    majorLength,
                                    minorWidth,
                                    majorAxisAngleDegrees,
                                    aspectRatio,
                                    contourFillRatio,
                                    hullCoverage,
                                    candidateCoverage,
                                    score
                            );

                    if (bestCandidate == null
                            || candidate.score
                            > bestCandidate.score) {
                        bestCandidate =
                                candidate;
                    }

                } finally {
                    contour2f.release();
                }
            }

            if (bestCandidate != null) {
                Log.d(
                        "PillMask",
                        "Lab-alapú kapszulafinomítás elfogadva. "
                                + "Lab FG/BG távolság="
                                + String.format(
                                Locale.ROOT,
                                "%.2f",
                                foregroundBackgroundDistance
                        )
                                + "; középpont=("
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.x
                        )
                                + ","
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.center.y
                        )
                                + "); hossz="
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.majorLength
                        )
                                + "; szélesség="
                                + String.format(
                                Locale.ROOT,
                                "%.1f",
                                bestCandidate.minorWidth
                        )
                );
            }

            return bestCandidate;

        } catch (Exception colorError) {
            Log.w(
                    "PillMask",
                    "A Lab-alapú kapszulafinomítás nem "
                            + "használható, a korábbi kapszulaág "
                            + "marad.",
                    colorError
            );

            return null;

        } finally {
            sourceMat.release();
            rgbMat.release();
            labMat.release();

            if (labRoi != null) {
                labRoi.release();
            }

            seedMask.release();
            borderMask.release();
            foregroundMask.release();
            hierarchy.release();
            openKernel.release();
            closeKernel.release();
            hullMask.release();
            candidateMask.release();
            intersectionMask.release();
            hullContour2f.release();

            for (MatOfPoint contour : contours) {
                contour.release();
            }
        }
    }

    private double[] calculateMaskedLabMedian(
            Mat labImage,
            Mat mask
    ) {
        if (labImage == null
                || labImage.empty()
                || labImage.type() != CvType.CV_8UC3
                || mask == null
                || mask.empty()
                || mask.rows() != labImage.rows()
                || mask.cols() != labImage.cols()) {
            return null;
        }

        int selectedPixelCount =
                Core.countNonZero(
                        mask
                );

        if (selectedPixelCount < 8) {
            return null;
        }

        byte[] labPixels =
                new byte[
                        labImage.rows()
                                * labImage.cols()
                                * 3
                        ];

        byte[] maskPixels =
                new byte[
                        mask.rows()
                                * mask.cols()
                        ];

        labImage.get(
                0,
                0,
                labPixels
        );

        mask.get(
                0,
                0,
                maskPixels
        );

        int[] lightnessValues =
                new int[
                        selectedPixelCount
                        ];

        int[] aValues =
                new int[
                        selectedPixelCount
                        ];

        int[] bValues =
                new int[
                        selectedPixelCount
                        ];

        int outputIndex =
                0;

        for (int pixelIndex = 0;
             pixelIndex < maskPixels.length;
             pixelIndex++) {
            if ((
                    maskPixels[pixelIndex]
                            & 0xFF
            ) == 0) {
                continue;
            }

            int labOffset =
                    pixelIndex * 3;

            lightnessValues[outputIndex] =
                    labPixels[
                            labOffset
                            ] & 0xFF;

            aValues[outputIndex] =
                    labPixels[
                            labOffset + 1
                            ] & 0xFF;

            bValues[outputIndex] =
                    labPixels[
                            labOffset + 2
                            ] & 0xFF;

            outputIndex++;
        }

        if (outputIndex < 8) {
            return null;
        }

        if (outputIndex
                != selectedPixelCount) {
            lightnessValues =
                    java.util.Arrays.copyOf(
                            lightnessValues,
                            outputIndex
                    );

            aValues =
                    java.util.Arrays.copyOf(
                            aValues,
                            outputIndex
                    );

            bValues =
                    java.util.Arrays.copyOf(
                            bValues,
                            outputIndex
                    );
        }

        java.util.Arrays.sort(
                lightnessValues
        );

        java.util.Arrays.sort(
                aValues
        );

        java.util.Arrays.sort(
                bValues
        );

        return new double[]{
                medianOfSortedIntArray(
                        lightnessValues
                ),
                medianOfSortedIntArray(
                        aValues
                ),
                medianOfSortedIntArray(
                        bValues
                )
        };
    }

    private double medianOfSortedIntArray(
            int[] sortedValues
    ) {
        if (sortedValues == null
                || sortedValues.length == 0) {
            return 0.0;
        }

        int middle =
                sortedValues.length / 2;

        if ((
                sortedValues.length
                        & 1
        ) == 1) {
            return sortedValues[
                    middle
                    ];
        }

        return (
                sortedValues[
                        middle - 1
                        ]
                        + sortedValues[
                        middle
                        ]
        ) * 0.5;
    }

    private double labDistance(
            double[] first,
            double[] second
    ) {
        if (first == null
                || second == null
                || first.length < 3
                || second.length < 3) {
            return Double.POSITIVE_INFINITY;
        }

        return labDistance(
                first[0],
                first[1],
                first[2],
                second
        );
    }

    private double labDistance(
            double lightness,
            double aChannel,
            double bChannel,
            double[] reference
    ) {
        if (reference == null
                || reference.length < 3) {
            return Double.POSITIVE_INFINITY;
        }

        double lightnessDifference =
                lightness
                        - reference[0];

        double aDifference =
                aChannel
                        - reference[1];

        double bDifference =
                bChannel
                        - reference[2];

        return Math.sqrt(
                lightnessDifference
                        * lightnessDifference
                        + aDifference
                        * aDifference
                        + bDifference
                        * bDifference
        );
    }

    private CapsuleShapeCandidate detectCapsuleShapeByBackgroundDistance(
            Bitmap sourceBitmap,
            GeminiEmbeddingService.PixelBox pixelBox,
            MatOfPoint hullContour,
            double hullArea,
            double geminiHullAspectRatio,
            int imageWidth,
            int imageHeight
    ) {
        if (sourceBitmap == null
                || sourceBitmap.isRecycled()
                || sourceBitmap.getWidth() != imageWidth
                || sourceBitmap.getHeight() != imageHeight
                || pixelBox == null
                || hullContour == null
                || hullContour.empty()
                || hullArea <= 0.0
                || geminiHullAspectRatio
                < CAPSULE_MIN_GEMINI_HULL_ASPECT_RATIO) {
            return null;
        }

        int boxLeft =
                Math.max(
                        0,
                        Math.min(
                                imageWidth - 1,
                                pixelBox.left
                        )
                );

        int boxTop =
                Math.max(
                        0,
                        Math.min(
                                imageHeight - 1,
                                pixelBox.top
                        )
                );

        int boxRight =
                Math.max(
                        boxLeft + 1,
                        Math.min(
                                imageWidth,
                                pixelBox.right
                        )
                );

        int boxBottom =
                Math.max(
                        boxTop + 1,
                        Math.min(
                                imageHeight,
                                pixelBox.bottom
                        )
                );

        int boxWidth =
                boxRight - boxLeft;

        int boxHeight =
                boxBottom - boxTop;

        int longerBoxSide =
                Math.max(
                        boxWidth,
                        boxHeight
                );

        if (longerBoxSide < 12) {
            return null;
        }

        int padding =
                Math.max(
                        5,
                        (int) Math.round(
                                longerBoxSide
                                        * CAPSULE_ROI_PADDING_RATIO
                        )
                );

        int roiLeft =
                Math.max(
                        0,
                        boxLeft - padding
                );

        int roiTop =
                Math.max(
                        0,
                        boxTop - padding
                );

        int roiRight =
                Math.min(
                        imageWidth,
                        boxRight + padding
                );

        int roiBottom =
                Math.min(
                        imageHeight,
                        boxBottom + padding
                );

        int roiWidth =
                roiRight - roiLeft;

        int roiHeight =
                roiBottom - roiTop;

        if (roiWidth < 16
                || roiHeight < 16) {
            return null;
        }

        Mat distanceImage =
                new Mat(
                        roiHeight,
                        roiWidth,
                        CvType.CV_8UC1
                );

        Mat foregroundMask =
                new Mat();

        Mat hierarchy =
                new Mat();

        Mat openKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                3,
                                3
                        )
                );

        Mat closeKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                5,
                                5
                        )
                );

        Mat hullMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat candidateMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        Mat intersectionMask =
                new Mat();

        List<MatOfPoint> contours =
                new ArrayList<>();

        try {
            int[] pixels =
                    new int[
                            roiWidth
                                    * roiHeight
                            ];

            sourceBitmap.getPixels(
                    pixels,
                    0,
                    roiWidth,
                    roiLeft,
                    roiTop,
                    roiWidth,
                    roiHeight
            );

            int borderThickness =
                    Math.max(
                            2,
                            (int) Math.round(
                                    Math.min(
                                            roiWidth,
                                            roiHeight
                                    ) * 0.10
                            )
                    );

            long redSum =
                    0L;

            long greenSum =
                    0L;

            long blueSum =
                    0L;

            int borderPixelCount =
                    0;

            for (int y = 0;
                 y < roiHeight;
                 y++) {
                for (int x = 0;
                     x < roiWidth;
                     x++) {
                    boolean borderPixel =
                            x < borderThickness
                                    || y < borderThickness
                                    || x >= roiWidth
                                    - borderThickness
                                    || y >= roiHeight
                                    - borderThickness;

                    if (!borderPixel) {
                        continue;
                    }

                    int color =
                            pixels[
                                    y * roiWidth
                                            + x
                                    ];

                    redSum +=
                            Color.red(
                                    color
                            );

                    greenSum +=
                            Color.green(
                                    color
                            );

                    blueSum +=
                            Color.blue(
                                    color
                            );

                    borderPixelCount++;
                }
            }

            if (borderPixelCount <= 0) {
                return null;
            }

            double backgroundRed =
                    redSum
                            / (double) borderPixelCount;

            double backgroundGreen =
                    greenSum
                            / (double) borderPixelCount;

            double backgroundBlue =
                    blueSum
                            / (double) borderPixelCount;

            byte[] distancePixels =
                    new byte[
                            roiWidth
                                    * roiHeight
                            ];

            for (int index = 0;
                 index < pixels.length;
                 index++) {
                int color =
                        pixels[index];

                double redDifference =
                        Color.red(
                                color
                        )
                                - backgroundRed;

                double greenDifference =
                        Color.green(
                                color
                        )
                                - backgroundGreen;

                double blueDifference =
                        Color.blue(
                                color
                        )
                                - backgroundBlue;

                double distance =
                        Math.sqrt(
                                0.30
                                        * redDifference
                                        * redDifference
                                        + 0.59
                                        * greenDifference
                                        * greenDifference
                                        + 0.11
                                        * blueDifference
                                        * blueDifference
                        );

                int roundedDistance =
                        Math.max(
                                0,
                                Math.min(
                                        255,
                                        (int) Math.round(
                                                distance
                                        )
                                )
                        );

                distancePixels[index] =
                        (byte) (
                                roundedDistance
                                        & 0xFF
                        );
            }

            distanceImage.put(
                    0,
                    0,
                    distancePixels
            );

            double otsuThreshold =
                    Imgproc.threshold(
                            distanceImage,
                            foregroundMask,
                            0.0,
                            255.0,
                            Imgproc.THRESH_BINARY
                                    | Imgproc.THRESH_OTSU
                    );

            double strictThreshold =
                    Math.min(
                            220.0,
                            Math.max(
                                    12.0,
                                    otsuThreshold
                                            + Math.max(
                                            8.0,
                                            otsuThreshold
                                                    * 0.30
                                    )
                            )
                    );

            Imgproc.threshold(
                    distanceImage,
                    foregroundMask,
                    strictThreshold,
                    255.0,
                    Imgproc.THRESH_BINARY
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
            );

            Imgproc.morphologyEx(
                    foregroundMask,
                    foregroundMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
            );

            Imgproc.findContours(
                    foregroundMask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
            );

            if (contours.isEmpty()) {
                return null;
            }

            Imgproc.fillPoly(
                    hullMask,
                    Collections.singletonList(
                            hullContour
                    ),
                    new Scalar(255)
            );

            double boxCenterX =
                    (
                            boxLeft
                                    + boxRight
                    ) * 0.5;

            double boxCenterY =
                    (
                            boxTop
                                    + boxBottom
                    ) * 0.5;

            double maximumCenterOffset =
                    longerBoxSide
                            * CAPSULE_MAX_CENTER_OFFSET_RATIO;

            double minimumContourArea =
                    Math.max(
                            20.0,
                            boxWidth
                                    * boxHeight
                                    * 0.08
                    );

            CapsuleShapeCandidate bestCandidate =
                    null;

            for (MatOfPoint contour : contours) {
                double contourArea =
                        Math.abs(
                                Imgproc.contourArea(
                                        contour
                                )
                        );

                if (contourArea
                        < minimumContourArea) {
                    continue;
                }

                MatOfPoint2f contour2f =
                        new MatOfPoint2f(
                                contour.toArray()
                        );

                try {
                    RotatedRect rectangle =
                            Imgproc.minAreaRect(
                                    contour2f
                            );

                    double rectangleWidth =
                            rectangle.size.width;

                    double rectangleHeight =
                            rectangle.size.height;

                    double minorWidth =
                            Math.min(
                                    rectangleWidth,
                                    rectangleHeight
                            );

                    double majorLength =
                            Math.max(
                                    rectangleWidth,
                                    rectangleHeight
                            );

                    if (minorWidth <= 1.0
                            || majorLength <= 1.0) {
                        continue;
                    }

                    double aspectRatio =
                            majorLength
                                    / minorWidth;

                    if (aspectRatio
                            < CAPSULE_MIN_IMAGE_ASPECT_RATIO
                            || aspectRatio
                            > CAPSULE_MAX_IMAGE_ASPECT_RATIO) {
                        continue;
                    }

                    double globalCenterX =
                            roiLeft
                                    + rectangle.center.x;

                    double globalCenterY =
                            roiTop
                                    + rectangle.center.y;

                    double centerDx =
                            globalCenterX
                                    - boxCenterX;

                    double centerDy =
                            globalCenterY
                                    - boxCenterY;

                    double centerOffset =
                            Math.sqrt(
                                    centerDx * centerDx
                                            + centerDy * centerDy
                            );

                    if (centerOffset
                            > maximumCenterOffset) {
                        continue;
                    }

                    double idealCapsuleArea =
                            (
                                    majorLength
                                            - minorWidth
                            ) * minorWidth
                                    + Math.PI
                                    * Math.pow(
                                    minorWidth
                                            * 0.5,
                                    2.0
                            );

                    double contourFillRatio =
                            contourArea
                                    / Math.max(
                                    1.0,
                                    idealCapsuleArea
                            );

                    if (contourFillRatio
                            < CAPSULE_MIN_CONTOUR_FILL_RATIO
                            || contourFillRatio
                            > CAPSULE_MAX_CONTOUR_FILL_RATIO) {
                        continue;
                    }

                    double majorAxisAngleDegrees =
                            rectangleWidth
                                    >= rectangleHeight
                                    ? rectangle.angle
                                    : rectangle.angle
                                      + 90.0;

                    Point globalCenter =
                            new Point(
                                    globalCenterX,
                                    globalCenterY
                            );

                    candidateMask.setTo(
                            new Scalar(0)
                    );

                    drawCapsuleMask(
                            candidateMask,
                            globalCenter,
                            majorLength,
                            minorWidth,
                            majorAxisAngleDegrees
                    );

                    Core.bitwise_and(
                            hullMask,
                            candidateMask,
                            intersectionMask
                    );

                    double intersectionArea =
                            Core.countNonZero(
                                    intersectionMask
                            );

                    double candidateArea =
                            Core.countNonZero(
                                    candidateMask
                            );

                    double hullCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    hullArea
                            );

                    double candidateCoverage =
                            intersectionArea
                                    / Math.max(
                                    1.0,
                                    candidateArea
                            );

                    if (hullCoverage
                            < CAPSULE_MIN_GEMINI_HULL_COVERAGE
                            || candidateCoverage
                            < CAPSULE_MIN_CANDIDATE_COVERAGE) {
                        continue;
                    }

                    double normalizedCenterOffset =
                            centerOffset
                                    / Math.max(
                                    1.0,
                                    maximumCenterOffset
                            );

                    double score =
                            hullCoverage * 4.0
                                    + candidateCoverage * 2.0
                                    + contourFillRatio * 2.0
                                    - normalizedCenterOffset;

                    CapsuleShapeCandidate candidate =
                            new CapsuleShapeCandidate(
                                    globalCenter,
                                    majorLength,
                                    minorWidth,
                                    majorAxisAngleDegrees,
                                    aspectRatio,
                                    contourFillRatio,
                                    hullCoverage,
                                    candidateCoverage,
                                    score
                            );

                    if (bestCandidate == null
                            || candidate.score
                            > bestCandidate.score) {
                        bestCandidate =
                                candidate;
                    }

                } finally {
                    contour2f.release();
                }
            }

            return bestCandidate;

        } catch (Exception capsuleError) {
            Log.w(
                    "PillMask",
                    "A kapszula OpenCV-finítása nem "
                            + "használható, marad a korábbi "
                            + "maszklogika.",
                    capsuleError
            );

            return null;

        } finally {
            distanceImage.release();
            foregroundMask.release();
            hierarchy.release();
            openKernel.release();
            closeKernel.release();
            hullMask.release();
            candidateMask.release();
            intersectionMask.release();

            for (MatOfPoint contour : contours) {
                contour.release();
            }
        }
    }

    private void drawCapsuleMask(
            Mat destination,
            Point center,
            double majorLength,
            double minorWidth,
            double majorAxisAngleDegrees
    ) {
        if (destination == null
                || destination.empty()
                || center == null
                || majorLength <= 1.0
                || minorWidth <= 1.0
                || majorLength < minorWidth) {
            return;
        }

        double radius =
                minorWidth
                        * 0.5;

        double halfCoreLength =
                Math.max(
                        0.0,
                        majorLength
                                * 0.5
                                - radius
                );

        double angleRadians =
                Math.toRadians(
                        majorAxisAngleDegrees
                );

        double directionX =
                Math.cos(
                        angleRadians
                );

        double directionY =
                Math.sin(
                        angleRadians
                );

        Point firstEnd =
                new Point(
                        center.x
                                - directionX
                                * halfCoreLength,
                        center.y
                                - directionY
                                * halfCoreLength
                );

        Point secondEnd =
                new Point(
                        center.x
                                + directionX
                                * halfCoreLength,
                        center.y
                                + directionY
                                * halfCoreLength
                );

        int lineThickness =
                Math.max(
                        1,
                        (int) Math.round(
                                minorWidth
                        )
                );

        int circleRadius =
                Math.max(
                        1,
                        (int) Math.round(
                                radius
                        )
                );

        Imgproc.line(
                destination,
                firstEnd,
                secondEnd,
                new Scalar(255),
                lineThickness,
                Imgproc.LINE_8
        );

        Imgproc.circle(
                destination,
                firstEnd,
                circleRadius,
                new Scalar(255),
                -1,
                Imgproc.LINE_8,
                0
        );

        Imgproc.circle(
                destination,
                secondEnd,
                circleRadius,
                new Scalar(255),
                -1,
                Imgproc.LINE_8,
                0
        );
    }

    private static final class CapsuleShapeCandidate {
        final Point center;
        final double majorLength;
        final double minorWidth;
        final double majorAxisAngleDegrees;
        final double aspectRatio;
        final double contourFillRatio;
        final double hullCoverage;
        final double candidateCoverage;
        final double score;

        CapsuleShapeCandidate(
                Point center,
                double majorLength,
                double minorWidth,
                double majorAxisAngleDegrees,
                double aspectRatio,
                double contourFillRatio,
                double hullCoverage,
                double candidateCoverage,
                double score
        ) {
            this.center =
                    center;

            this.majorLength =
                    majorLength;

            this.minorWidth =
                    minorWidth;

            this.majorAxisAngleDegrees =
                    majorAxisAngleDegrees;

            this.aspectRatio =
                    aspectRatio;

            this.contourFillRatio =
                    contourFillRatio;

            this.hullCoverage =
                    hullCoverage;

            this.candidateCoverage =
                    candidateCoverage;

            this.score =
                    score;
        }
    }

    private double pointDistance(
            Point first,
            Point second
    ) {
        if (first == null || second == null) {
            return Double.POSITIVE_INFINITY;
        }

        double dx = first.x - second.x;
        double dy = first.y - second.y;

        return Math.sqrt(dx * dx + dy * dy);
    }

    private Bitmap maskToBitmap(Mat mask) throws Exception {
        if (mask == null || mask.empty()) {
            throw new Exception("A bináris maszk üres.");
        }

        Bitmap bitmap = Bitmap.createBitmap(
                mask.cols(),
                mask.rows(),
                Bitmap.Config.ARGB_8888
        );

        Utils.matToBitmap(mask, bitmap);
        return bitmap;
    }

    private Bitmap createBlueSegmentationOverlay(
            Bitmap source,
            Mat binaryMask,
            GeminiEmbeddingService.SegmentationResult segmentation
    ) throws Exception {
        Mat image = new Mat();
        Mat blueImage = new Mat();
        Mat blended = new Mat();
        Mat result = new Mat();
        Mat maskForContours = new Mat();
        Mat hierarchy = new Mat();

        List<MatOfPoint> actualMaskContours =
                new ArrayList<>();

        try {
            Utils.bitmapToMat(source, image);
            image.copyTo(blueImage);

            blueImage.setTo(
                    new Scalar(0, 0, 255, 255),
                    binaryMask
            );

            Core.addWeighted(
                    image,
                    0.45,
                    blueImage,
                    0.55,
                    0.0,
                    blended
            );

            image.copyTo(result);
            blended.copyTo(result, binaryMask);

            GeminiEmbeddingService.PixelBox box =
                    segmentation.getPixelBox();

            Imgproc.rectangle(
                    result,
                    new Point(box.left, box.top),
                    new Point(box.right - 1, box.bottom - 1),
                    new Scalar(255, 0, 0, 255),
                    3
            );

            binaryMask.copyTo(maskForContours);

            Imgproc.findContours(
                    maskForContours,
                    actualMaskContours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
            );

            Imgproc.drawContours(
                    result,
                    actualMaskContours,
                    -1,
                    new Scalar(0, 255, 0, 255),
                    3
            );

            Imgproc.putText(
                    result,
                    segmentation.getLabel(),
                    new Point(
                            box.left,
                            Math.max(25, box.top - 8)
                    ),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    new Scalar(255, 0, 0, 255),
                    2
            );

            Bitmap output = Bitmap.createBitmap(
                    result.cols(),
                    result.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(result, output);
            return output;

        } finally {
            image.release();
            blueImage.release();
            blended.release();
            result.release();
            maskForContours.release();
            hierarchy.release();

            for (MatOfPoint contour : actualMaskContours) {
                if (contour != null) {
                    contour.release();
                }
            }

            actualMaskContours.clear();
        }
    }

    private Bitmap createGrayMaskedBitmap(
            Bitmap source,
            Mat binaryMask
    ) throws Exception {
        Bitmap output = source.copy(Bitmap.Config.ARGB_8888, true);
        Mat image = new Mat();
        Mat inverseMask = new Mat();

        try {
            Utils.bitmapToMat(output, image);
            Core.bitwise_not(binaryMask, inverseMask);

            image.setTo(
                    new Scalar(
                            SEGMENTED_BACKGROUND_GRAY,
                            SEGMENTED_BACKGROUND_GRAY,
                            SEGMENTED_BACKGROUND_GRAY,
                            255
                    ),
                    inverseMask
            );

            Utils.matToBitmap(image, output);
            return output;
        } finally {
            image.release();
            inverseMask.release();
        }
    }

    private Mat createColoredEmbeddingRenderMask(
            Bitmap grayMaskedBitmap,
            Mat binaryMask
    ) {
        Mat fallbackMask =
                binaryMask.clone();

        if (grayMaskedBitmap == null
                || grayMaskedBitmap.isRecycled()
                || binaryMask == null
                || binaryMask.empty()
                || grayMaskedBitmap.getWidth() != binaryMask.cols()
                || grayMaskedBitmap.getHeight() != binaryMask.rows()) {
            return fallbackMask;
        }

        Mat sourceMat =
                new Mat();

        Mat rgbMat =
                new Mat();

        Mat hsvMat =
                new Mat();

        Mat seedMask =
                new Mat();

        Mat colorMask =
                new Mat();

        Mat candidateMask =
                new Mat();

        Mat contourInput =
                new Mat();

        Mat selectedMask =
                Mat.zeros(
                        binaryMask.size(),
                        CvType.CV_8UC1
                );

        Mat componentMask =
                Mat.zeros(
                        binaryMask.size(),
                        CvType.CV_8UC1
                );

        Mat overlapMask =
                new Mat();

        MatOfPoint nonZeroPoints =
                new MatOfPoint();

        Mat seedKernel =
                null;

        Mat openKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                5,
                                5
                        )
                );

        Mat closeKernel =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new org.opencv.core.Size(
                                5,
                                5
                        )
                );

        Mat hierarchy =
                new Mat();

        List<MatOfPoint> contours =
                new ArrayList<>();

        try {
            Core.findNonZero(
                    binaryMask,
                    nonZeroPoints
            );

            if (nonZeroPoints.empty()) {
                return fallbackMask;
            }

            org.opencv.core.Rect originalBounds =
                    Imgproc.boundingRect(
                            nonZeroPoints
                    );

            int originalShortSide =
                    Math.min(
                            originalBounds.width,
                            originalBounds.height
                    );

            int originalLongSide =
                    Math.max(
                            originalBounds.width,
                            originalBounds.height
                    );

            if (originalShortSide <= 0
                    || originalLongSide
                    / (double) originalShortSide
                    < 1.55) {
                return fallbackMask;
            }

            int seedKernelSize =
                    Math.max(
                            3,
                            (int) Math.round(
                                    originalShortSide
                                            * 0.18
                            )
                    );

            if ((seedKernelSize & 1) == 0) {
                seedKernelSize++;
            }

            seedKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new org.opencv.core.Size(
                                    seedKernelSize,
                                    seedKernelSize
                            )
                    );

            Imgproc.erode(
                    binaryMask,
                    seedMask,
                    seedKernel
            );

            if (Core.countNonZero(seedMask) < 20) {
                return fallbackMask;
            }

            Utils.bitmapToMat(
                    grayMaskedBitmap,
                    sourceMat
            );

            if (sourceMat.channels() == 4) {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_RGBA2RGB
                );
            } else if (sourceMat.channels() == 3) {
                sourceMat.copyTo(
                        rgbMat
                );
            } else {
                Imgproc.cvtColor(
                        sourceMat,
                        rgbMat,
                        Imgproc.COLOR_GRAY2RGB
                );
            }

            Imgproc.cvtColor(
                    rgbMat,
                    hsvMat,
                    Imgproc.COLOR_RGB2HSV
            );

            Scalar seedMean =
                    Core.mean(
                            hsvMat,
                            seedMask
                    );

            double seedSaturation =
                    seedMean.val[1];

            double seedValue =
                    seedMean.val[2];

            if (seedSaturation < 60.0) {
                return fallbackMask;
            }

            int minimumSaturation =
                    Math.max(
                            45,
                            (int) Math.round(
                                    seedSaturation
                                            * 0.30
                            )
                    );

            int minimumValue =
                    Math.max(
                            15,
                            (int) Math.round(
                                    seedValue
                                            * 0.12
                            )
                    );

            Core.inRange(
                    hsvMat,
                    new Scalar(
                            0,
                            minimumSaturation,
                            minimumValue
                    ),
                    new Scalar(
                            179,
                            255,
                            255
                    ),
                    colorMask
            );

            Core.bitwise_and(
                    colorMask,
                    binaryMask,
                    candidateMask
            );

            Imgproc.morphologyEx(
                    candidateMask,
                    candidateMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
            );

            Imgproc.morphologyEx(
                    candidateMask,
                    candidateMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
            );

            candidateMask.copyTo(
                    contourInput
            );

            Imgproc.findContours(
                    contourInput,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            if (contours.isEmpty()) {
                return fallbackMask;
            }

            double bestScore =
                    -1.0;

            for (MatOfPoint contour : contours) {
                if (contour == null
                        || contour.empty()) {
                    continue;
                }

                double contourArea =
                        Math.abs(
                                Imgproc.contourArea(
                                        contour
                                )
                        );

                if (contourArea < 20.0) {
                    continue;
                }

                componentMask.setTo(
                        new Scalar(0)
                );

                Imgproc.drawContours(
                        componentMask,
                        Collections.singletonList(
                                contour
                        ),
                        0,
                        new Scalar(255),
                        -1,
                        Imgproc.LINE_8
                );

                Core.bitwise_and(
                        componentMask,
                        seedMask,
                        overlapMask
                );

                int seedOverlap =
                        Core.countNonZero(
                                overlapMask
                        );

                double score =
                        seedOverlap * 10.0
                                + contourArea;

                if (score > bestScore) {
                    bestScore =
                            score;

                    componentMask.copyTo(
                            selectedMask
                    );
                }
            }

            int originalArea =
                    Core.countNonZero(
                            binaryMask
                    );

            int selectedArea =
                    Core.countNonZero(
                            selectedMask
                    );

            if (selectedArea <= 0
                    || selectedArea
                    / (double) Math.max(
                    1,
                    originalArea
            )
                    < 0.50) {
                return fallbackMask;
            }

            Core.bitwise_and(
                    selectedMask,
                    binaryMask,
                    selectedMask
            );

            fallbackMask.release();

            Log.d(
                    "PillEmbeddingRender",
                    "Színes gyógyszer árnyékmentesítve. "
                            + "seedSaturation="
                            + String.format(
                            Locale.ROOT,
                            "%.1f",
                            seedSaturation
                    )
                            + "; minimumSaturation="
                            + minimumSaturation
                            + "; minimumValue="
                            + minimumValue
            );

            return selectedMask.clone();

        } catch (Exception error) {
            Log.e(
                    "PillEmbeddingRender",
                    "Az embeddingkép árnyékmentesítése sikertelen.",
                    error
            );

            return fallbackMask;

        } finally {
            sourceMat.release();
            rgbMat.release();
            hsvMat.release();
            seedMask.release();
            colorMask.release();
            candidateMask.release();
            contourInput.release();
            selectedMask.release();
            componentMask.release();
            overlapMask.release();
            nonZeroPoints.release();

            if (seedKernel != null) {
                seedKernel.release();
            }

            openKernel.release();
            closeKernel.release();
            hierarchy.release();

            for (MatOfPoint contour : contours) {
                if (contour != null) {
                    contour.release();
                }
            }

            contours.clear();
        }
    }

    private Bitmap createSquareEmbeddingInput(
            Bitmap grayMaskedBitmap,
            Mat binaryMask
    ) throws Exception {
        Mat renderMask =
                createColoredEmbeddingRenderMask(
                        grayMaskedBitmap,
                        binaryMask
                );

        MatOfPoint nonZeroPoints =
                new MatOfPoint();

        Bitmap cleanedBitmap =
                null;

        Bitmap crop =
                null;

        Bitmap scaled =
                null;

        try {
            Core.findNonZero(
                    renderMask,
                    nonZeroPoints
            );

            if (nonZeroPoints.empty()) {
                throw new Exception(
                        "A bináris maszk nem tartalmaz gyógyszert."
                );
            }

            cleanedBitmap =
                    createGrayMaskedBitmap(
                            grayMaskedBitmap,
                            renderMask
                    );

            org.opencv.core.Rect bounds =
                    Imgproc.boundingRect(
                            nonZeroPoints
                    );

            int padding =
                    Math.max(
                            6,
                            Math.round(
                                    Math.max(
                                            bounds.width,
                                            bounds.height
                                    ) * 0.12f
                            )
                    );

            int left =
                    clamp(
                            bounds.x - padding,
                            0,
                            cleanedBitmap.getWidth() - 1
                    );

            int top =
                    clamp(
                            bounds.y - padding,
                            0,
                            cleanedBitmap.getHeight() - 1
                    );

            int right =
                    clamp(
                            bounds.x
                                    + bounds.width
                                    + padding,
                            left + 1,
                            cleanedBitmap.getWidth()
                    );

            int bottom =
                    clamp(
                            bounds.y
                                    + bounds.height
                                    + padding,
                            top + 1,
                            cleanedBitmap.getHeight()
                    );

            crop =
                    Bitmap.createBitmap(
                            cleanedBitmap,
                            left,
                            top,
                            right - left,
                            bottom - top
                    );

            final int outputSize =
                    512;

            int medicineLongestSide =
                    Math.max(
                            bounds.width,
                            bounds.height
                    );

            int medicineShortestSide =
                    Math.max(
                            1,
                            Math.min(
                                    bounds.width,
                                    bounds.height
                            )
                    );

            double medicineAspectRatio =
                    medicineLongestSide
                            / (double) medicineShortestSide;

            double normalizedAspectRatio =
                    (
                            medicineAspectRatio
                                    - REFERENCE_ROUND_MEDICINE_ASPECT_RATIO
                    )
                            / (
                            REFERENCE_ELONGATED_MEDICINE_ASPECT_RATIO
                                    - REFERENCE_ROUND_MEDICINE_ASPECT_RATIO
                    );

            normalizedAspectRatio =
                    Math.max(
                            0.0,
                            Math.min(
                                    1.0,
                                    normalizedAspectRatio
                            )
                    );

            int targetMedicineLongestSide =
                    (int) Math.round(
                            REFERENCE_ROUND_MEDICINE_LONGEST_SIDE_PX
                                    + normalizedAspectRatio
                                    * (
                                    REFERENCE_ELONGATED_MEDICINE_LONGEST_SIDE_PX
                                            - REFERENCE_ROUND_MEDICINE_LONGEST_SIDE_PX
                            )
                    );

            float scale =
                    targetMedicineLongestSide
                            / (float) medicineLongestSide;

            Log.d(
                    "PillEmbeddingScale",
                    "aspect="
                            + String.format(
                            Locale.ROOT,
                            "%.3f",
                            medicineAspectRatio
                    )
                            + "; targetLongestSide="
                            + targetMedicineLongestSide
            );

            int scaledWidth =
                    Math.max(
                            1,
                            Math.round(
                                    crop.getWidth()
                                            * scale
                            )
                    );

            int scaledHeight =
                    Math.max(
                            1,
                            Math.round(
                                    crop.getHeight()
                                            * scale
                            )
                    );

            scaled =
                    Bitmap.createScaledBitmap(
                            crop,
                            scaledWidth,
                            scaledHeight,
                            true
                    );

            Bitmap output =
                    Bitmap.createBitmap(
                            outputSize,
                            outputSize,
                            Bitmap.Config.ARGB_8888
                    );

            Canvas canvas =
                    new Canvas(
                            output
                    );

            canvas.drawColor(
                    Color.rgb(
                            SEGMENTED_BACKGROUND_GRAY,
                            SEGMENTED_BACKGROUND_GRAY,
                            SEGMENTED_BACKGROUND_GRAY
                    )
            );

            Paint paint =
                    new Paint(
                            Paint.ANTI_ALIAS_FLAG
                                    | Paint.FILTER_BITMAP_FLAG
                    );

            float drawX =
                    (outputSize - scaledWidth)
                            / 2f;

            float drawY =
                    (outputSize - scaledHeight)
                            / 2f;

            canvas.drawBitmap(
                    scaled,
                    drawX,
                    drawY,
                    paint
            );

            return output;

        } finally {
            renderMask.release();
            nonZeroPoints.release();

            if (scaled != null
                    && scaled != crop
                    && !scaled.isRecycled()) {
                scaled.recycle();
            }

            if (crop != null
                    && !crop.isRecycled()) {
                crop.recycle();
            }

            if (cleanedBitmap != null
                    && !cleanedBitmap.isRecycled()) {
                cleanedBitmap.recycle();
            }
        }
    }

    private Bitmap cropMedicineSearchArea(
            Bitmap correctedBitmap
    ) throws Exception {
        if (correctedBitmap == null
                || correctedBitmap.getWidth() <= 0
                || correctedBitmap.getHeight() <= 0) {
            throw new Exception(
                    "A korrigált gyógyszermező képe hibás."
            );
        }

        double pixelsPerMillimeter =
                ArUcoProcessor.getPixelsPerMillimeter();

        if (pixelsPerMillimeter <= 0.0
                || Double.isNaN(pixelsPerMillimeter)
                || Double.isInfinite(pixelsPerMillimeter)) {
            throw new Exception(
                    "Az ArUco-mező pixel/mm aránya hibás: "
                            + pixelsPerMillimeter
            );
        }

        int cropWidth =
                Math.min(
                        correctedBitmap.getWidth(),
                        Math.max(
                                1,
                                (int) Math.round(
                                        SEGMENT_SEARCH_WIDTH_MM
                                                * pixelsPerMillimeter
                                )
                        )
                );

        int cropHeight =
                Math.min(
                        correctedBitmap.getHeight(),
                        Math.max(
                                1,
                                (int) Math.round(
                                        SEGMENT_SEARCH_HEIGHT_MM
                                                * pixelsPerMillimeter
                                )
                        )
                );

        int centeredLeft =
                (
                        correctedBitmap.getWidth()
                                - cropWidth
                ) / 2;

        int centeredTop =
                (
                        correctedBitmap.getHeight()
                                - cropHeight
                ) / 2;

        int verticalOffsetPixels =
                (int) Math.round(
                        SEGMENT_SEARCH_CENTER_OFFSET_Y_MM
                                * pixelsPerMillimeter
                );

        int left =
                clamp(
                        centeredLeft,
                        0,
                        correctedBitmap.getWidth()
                                - cropWidth
                );

        int top =
                clamp(
                        centeredTop
                                + verticalOffsetPixels,
                        0,
                        correctedBitmap.getHeight()
                                - cropHeight
                );

        Log.d(
                "PillCrop",
                "Középre igazított keresési terület: corrected="
                        + correctedBitmap.getWidth()
                        + "x"
                        + correctedBitmap.getHeight()
                        + "; crop="
                        + cropWidth
                        + "x"
                        + cropHeight
                        + "; left="
                        + left
                        + "; top="
                        + top
                        + "; offsetYmm="
                        + String.format(
                        Locale.ROOT,
                        "%.2f",
                        SEGMENT_SEARCH_CENTER_OFFSET_Y_MM
                )
                        + "; px/mm="
                        + String.format(
                        Locale.ROOT,
                        "%.4f",
                        pixelsPerMillimeter
                )
        );

        return Bitmap.createBitmap(
                correctedBitmap,
                left,
                top,
                cropWidth,
                cropHeight
        );
    }

    private int getDatabaseEmbeddingDimension(Context context) throws Exception {
        DatabaseManager db = null;
        Cursor cursor = null;

        try {
            db = new DatabaseManager(context);
            cursor = db.getAllMedications();

            if (cursor == null || cursor.getCount() == 0) {
                throw new Exception("Az embedding-adatbázis üres.");
            }

            while (cursor.moveToNext()) {
                float[] left = safeVectorFromString(cursor.getString(1));
                if (left.length > 0) {
                    return left.length;
                }

                float[] right = safeVectorFromString(cursor.getString(2));
                if (right.length > 0) {
                    return right.length;
                }
            }

            throw new Exception(
                    "Az adatbázisban nincs használható embedding-vektor."
            );
        } finally {
            if (cursor != null) cursor.close();
            if (db != null) db.close();
        }
    }

    private void saveDebugBitmapQuietly(
            ProcessingDebugSaver saver,
            String filename,
            Bitmap bitmap
    ) {
        try {
            String normalizedFilename =
                    filename == null
                            ? ""
                            : filename.toLowerCase(Locale.ROOT);

            if (normalizedFilename.endsWith(".png")) {
                saver.saveBitmapPng(
                        filename,
                        bitmap
                );
            } else {
                saver.saveBitmapJpeg(
                        filename,
                        bitmap,
                        95
                );
            }

            showSavedFilename(filename);

        } catch (Exception e) {
            Log.w(
                    "PillDebug",
                    "Nem menthető: " + filename,
                    e
            );
        }
    }

    private void saveDebugTextQuietly(
            ProcessingDebugSaver saver,
            String filename,
            String content,
            String mimeType
    ) {
        try {
            saver.saveText(filename, content, mimeType);
        } catch (Exception e) {
            Log.w("PillDebug", "Nem menthető: " + filename, e);
        }
    }

    private void setProcessingControlsEnabled(boolean enabled) {
        handler.post(() -> {
            if (!isAdded()) return;

            if (btnOpenCamera != null) {
                btnOpenCamera.setEnabled(enabled);
            }

            if (btnOpenGallery != null) {
                btnOpenGallery.setEnabled(enabled);
            }

            boolean editorEnabled =
                    enabled
                            && hasEditablePillPhoto;

            if (sliderContrast != null) {
                sliderContrast.setEnabled(
                        editorEnabled
                );
            }

            if (btnFlash != null) {
                btnFlash.setEnabled(
                        editorEnabled
                );
            }
        });
    }

    private ParallelRecognitionResult recognizeMedicineInParallel(
            Context appContext,
            GeminiEmbeddingService service,
            Bitmap embeddingInputBitmap,
            int databaseDimension,
            int displayedIndex,
            GeminiEmbeddingService.SegmentationResult segmentation
    ) {
        long embeddingDurationMs = 0L;
        long databaseDurationMs = 0L;

        try {
            Log.d(
                    "PillParallel",
                    "Párhuzamos embedding indul: "
                            + displayedIndex
            );

            long embeddingStartedAt =
                    SystemClock.elapsedRealtime();

            float[] queryVector;

            try {
                queryVector =
                        service.createEmbedding(
                                embeddingInputBitmap,
                                databaseDimension
                        );
            } finally {
                embeddingDurationMs =
                        SystemClock.elapsedRealtime()
                                - embeddingStartedAt;
            }

            long databaseStartedAt =
                    SystemClock.elapsedRealtime();

            MatchResult matchResult;

            try {
                matchResult =
                        findBestMatch(
                                appContext,
                                queryVector
                        );
            } finally {
                databaseDurationMs =
                        SystemClock.elapsedRealtime()
                                - databaseStartedAt;
            }

            Log.d(
                    "PillParallel",
                    "Párhuzamos felismerés kész: "
                            + displayedIndex
                            + "; embedding="
                            + formatSeconds(
                            millisecondsToSeconds(
                                    embeddingDurationMs
                            )
                    )
                            + " s; db="
                            + formatSeconds(
                            millisecondsToSeconds(
                                    databaseDurationMs
                            )
                    )
                            + " s"
            );

            return ParallelRecognitionResult.success(
                    displayedIndex,
                    segmentation,
                    matchResult,
                    queryVector.length,
                    embeddingDurationMs,
                    databaseDurationMs
            );

        } catch (Exception error) {
            Log.e(
                    "PillParallel",
                    "A(z) "
                            + displayedIndex
                            + ". gyógyszer párhuzamos "
                            + "felismerése sikertelen.",
                    error
            );

            return ParallelRecognitionResult.failure(
                    displayedIndex,
                    segmentation,
                    embeddingDurationMs,
                    databaseDurationMs,
                    error
            );

        } finally {
            if (embeddingInputBitmap != null
                    && !embeddingInputBitmap.isRecycled()) {
                embeddingInputBitmap.recycle();
            }
        }
    }

    private void startProcessing(
            Bitmap bitmap,
            Bitmap originalBeforeFilters
    ) {
        if (bitmap == null
                || originalBeforeFilters == null) {
            return;
        }

        if (isProcessing) {
            showError("A képfeldolgozás már folyamatban van.");
            return;
        }

        isProcessing = true;
        setProcessingControlsEnabled(false);

        final Uri savedOriginalImageUri =
                saveBitmapToGallery(
                        originalBeforeFilters
                );

        tvResultName.setText(
                BuildConfig.DEBUG
                        ? "ArUco-korrekció folyamatban..."
                        : "A lefényképezett lap igazítása folyamatban..."
        );

        Context appContext = requireContext().getApplicationContext();
        ProcessingDebugSaver debugSaver =
                new ProcessingDebugSaver(appContext);

        final String evaluationFilename =
                currentEvaluationFilename == null
                        ? ""
                        : currentEvaluationFilename;

        executor.execute(() -> {
            long totalStart = SystemClock.elapsedRealtime();

            long arucoMs = 0L;
            long cropMs = 0L;
            long segmentationMs = 0L;
            long maskMs = 0L;
            long embeddingPreparationMs = 0L;
            long embeddingMs = 0L;
            long databaseMs = 0L;
            long parallelRecognitionWallMs = 0L;

            Mat inputMat = null;
            Mat processedMat = null;

            Bitmap correctedBitmap = null;
            Bitmap searchAreaBitmap = null;
            Bitmap combinedOverlayBitmap = null;
            Bitmap singleResultPreviewBitmap = null;

            try {
                saveDebugBitmapQuietly(
                        debugSaver,
                        "01_original.jpg",
                        originalBeforeFilters
                );

                saveDebugBitmapQuietly(
                        debugSaver,
                        "01a_contrast_brightness_input.jpg",
                        bitmap
                );

                inputMat = new Mat();
                Utils.bitmapToMat(bitmap, inputMat);

                long stepStart = SystemClock.elapsedRealtime();
                processedMat = ArUcoProcessor.processSheet(inputMat);
                arucoMs = SystemClock.elapsedRealtime() - stepStart;

                if (processedMat == null || processedMat.empty()) {
                    throw new Exception(
                            "Az ArUco-korrekció nem adott használható képet."
                    );
                }

                correctedBitmap = Bitmap.createBitmap(
                        processedMat.cols(),
                        processedMat.rows(),
                        Bitmap.Config.ARGB_8888
                );
                Utils.matToBitmap(processedMat, correctedBitmap);

                saveDebugBitmapQuietly(
                        debugSaver,
                        "02_aruco_corrected.jpg",
                        correctedBitmap
                );

                stepStart = SystemClock.elapsedRealtime();
                searchAreaBitmap = cropMedicineSearchArea(correctedBitmap);
                cropMs = SystemClock.elapsedRealtime() - stepStart;

                saveDebugBitmapQuietly(
                        debugSaver,
                        "03_search_area.jpg",
                        searchAreaBitmap
                );

                Bitmap searchPreview = searchAreaBitmap.copy(
                        Bitmap.Config.ARGB_8888,
                        false
                );

                handler.post(() -> {
                    if (!isAdded()
                            || ivSelectedPill == null
                            || tvResultName == null) {
                        return;
                    }

                    ivSelectedPill.setScaleType(ImageView.ScaleType.FIT_CENTER);
                    ivSelectedPill.setImageBitmap(searchPreview);
                    tvResultName.setText(
                            BuildConfig.DEBUG
                                    ? "Gyógyszerek szegmentálása "
                                      + "(Gemini 3.6 Flash)..."
                                    : "A gyógyszerek elkülönítése "
                                      + "folyamatban..."
                    );
                });

                GeminiEmbeddingService service =
                        new GeminiEmbeddingService();

                showProcessingStatus(
                        "Képelemzés folyamatban."
                );

                stepStart = SystemClock.elapsedRealtime();

                List<GeminiEmbeddingService.SegmentationResult>
                        segmentations =
                        service.segmentMedicines(searchAreaBitmap);

                segmentationMs =
                        SystemClock.elapsedRealtime() - stepStart;

                if (segmentations == null
                        || segmentations.isEmpty()) {
                    throw new Exception(
                            "A Gemini nem talált feldolgozható gyógyszert."
                    );
                }

                GeminiEmbeddingService.SegmentationResult
                        firstSegmentation =
                        segmentations.get(0);

                saveDebugTextQuietly(
                        debugSaver,
                        "04_segmentation_response.json",
                        firstSegmentation.getStructuredJson(),
                        "application/json"
                );

                saveDebugTextQuietly(
                        debugSaver,
                        "04a_interactions_raw_response.json",
                        firstSegmentation.getRawApiResponse(),
                        "application/json"
                );

                final int medicineCount =
                        segmentations.size();

                int databaseDimension =
                        getDatabaseEmbeddingDimension(appContext);

                final String expectedMedicineName =
                        resolveExpectedMedicineNameFromFilename(
                                appContext,
                                evaluationFilename
                        );

                final int shownDimension =
                        databaseDimension;

                List<MatchResult> results =
                        new ArrayList<>();

                JSONArray resultItems =
                        new JSONArray();

                combinedOverlayBitmap =
                        searchAreaBitmap.copy(
                                Bitmap.Config.ARGB_8888,
                                true
                        );

                List<ParallelRecognitionResult>
                        parallelResults =
                        new ArrayList<>(
                                Collections.nCopies(
                                        medicineCount,
                                        null
                                )
                        );

                List<Future<ParallelRecognitionResult>>
                        recognitionFutures =
                        new ArrayList<>();

                long firstRecognitionSubmittedAt = -1L;

                for (int index = 0;
                     index < medicineCount;
                     index++) {

                    GeminiEmbeddingService.SegmentationResult
                            segmentation =
                            segmentations.get(index);

                    final int displayedIndex =
                            index + 1;

                    Mat binaryMask = null;
                    Bitmap maskBitmap = null;
                    Bitmap grayMaskedBitmap = null;
                    Bitmap embeddingInputBitmap = null;

                    try {
                        showProcessingStatus(
                                "Gyógyszer előkészítése: "
                                        + displayedIndex
                                        + "/"
                                        + medicineCount
                        );

                        stepStart =
                                SystemClock.elapsedRealtime();

                        binaryMask = createPillMask(
                                searchAreaBitmap.getWidth(),
                                searchAreaBitmap.getHeight(),
                                segmentation.getPolygonPixels(),
                                searchAreaBitmap,
                                segmentation.getPixelBox()
                        );

                        maskBitmap =
                                maskToBitmap(binaryMask);

                        Bitmap nextCombinedOverlay =
                                createBlueSegmentationOverlay(
                                        combinedOverlayBitmap,
                                        binaryMask,
                                        segmentation
                                );

                        if (combinedOverlayBitmap != null
                                && combinedOverlayBitmap
                                != searchAreaBitmap
                                && !combinedOverlayBitmap.isRecycled()) {
                            combinedOverlayBitmap.recycle();
                        }

                        combinedOverlayBitmap =
                                nextCombinedOverlay;

                        grayMaskedBitmap =
                                createGrayMaskedBitmap(
                                        searchAreaBitmap,
                                        binaryMask
                                );

                        maskMs +=
                                SystemClock.elapsedRealtime()
                                        - stepStart;

                        String maskFilename =
                                medicineCount == 1
                                        ? "05_binary_mask.png"
                                        : String.format(
                                        Locale.ROOT,
                                        "05_pill_%02d_binary_mask.png",
                                        displayedIndex
                                );

                        String overlayFilename =
                                medicineCount == 1
                                        ? "06_blue_segmentation_overlay.jpg"
                                        : String.format(
                                        Locale.ROOT,
                                        "06_pill_%02d_blue_overlay.jpg",
                                        displayedIndex
                                );

                        String grayFilename =
                                medicineCount == 1
                                        ? "07_masked_gray_background.jpg"
                                        : String.format(
                                        Locale.ROOT,
                                        "07_pill_%02d_masked_gray.jpg",
                                        displayedIndex
                                );

                        saveDebugBitmapQuietly(
                                debugSaver,
                                maskFilename,
                                maskBitmap
                        );

                        saveDebugBitmapQuietly(
                                debugSaver,
                                overlayFilename,
                                combinedOverlayBitmap
                        );

                        saveDebugBitmapQuietly(
                                debugSaver,
                                grayFilename,
                                grayMaskedBitmap
                        );

                        stepStart =
                                SystemClock.elapsedRealtime();

                        embeddingInputBitmap =
                                createSquareEmbeddingInput(
                                        grayMaskedBitmap,
                                        binaryMask
                                );

                        embeddingPreparationMs +=
                                SystemClock.elapsedRealtime()
                                        - stepStart;

                        String embeddingFilename =
                                medicineCount == 1
                                        ? "08_embedding_input.jpg"
                                        : String.format(
                                        Locale.ROOT,
                                        "08_pill_%02d_embedding_input.jpg",
                                        displayedIndex
                                );

                        saveDebugBitmapQuietly(
                                debugSaver,
                                embeddingFilename,
                                embeddingInputBitmap
                        );

                        if (medicineCount == 1) {
                            singleResultPreviewBitmap =
                                    embeddingInputBitmap.copy(
                                            Bitmap.Config.ARGB_8888,
                                            false
                                    );
                        }

                        final Bitmap recognitionBitmap =
                                embeddingInputBitmap;

                        embeddingInputBitmap = null;

                        final GeminiEmbeddingService
                                .SegmentationResult
                                recognitionSegmentation =
                                segmentation;

                        if (firstRecognitionSubmittedAt < 0L) {
                            firstRecognitionSubmittedAt =
                                    SystemClock.elapsedRealtime();
                        }

                        Future<ParallelRecognitionResult>
                                recognitionFuture =
                                recognitionExecutor.submit(
                                        () ->
                                                recognizeMedicineInParallel(
                                                        appContext,
                                                        service,
                                                        recognitionBitmap,
                                                        databaseDimension,
                                                        displayedIndex,
                                                        recognitionSegmentation
                                                )
                                );

                        recognitionFutures.add(
                                recognitionFuture
                        );

                    } catch (Exception preparationError) {
                        Log.e(
                                "PillScan",
                                "A(z) "
                                        + displayedIndex
                                        + ". gyógyszer "
                                        + "előkészítése sikertelen.",
                                preparationError
                        );

                        parallelResults.set(
                                index,
                                ParallelRecognitionResult.failure(
                                        displayedIndex,
                                        segmentation,
                                        0L,
                                        0L,
                                        preparationError
                                )
                        );

                    } finally {
                        if (binaryMask != null) {
                            binaryMask.release();
                        }

                        if (maskBitmap != null
                                && !maskBitmap.isRecycled()) {
                            maskBitmap.recycle();
                        }

                        if (grayMaskedBitmap != null
                                && !grayMaskedBitmap.isRecycled()) {
                            grayMaskedBitmap.recycle();
                        }

                        if (embeddingInputBitmap != null
                                && !embeddingInputBitmap.isRecycled()) {
                            embeddingInputBitmap.recycle();
                        }
                    }
                }

                if (!recognitionFutures.isEmpty()) {
                    showProcessingStatus(
                            BuildConfig.DEBUG
                                    ? "Gyógyszerek párhuzamos felismerése "
                                      + recognitionFutures.size()
                                      + " feladattal..."
                                    : "A gyógyszerek azonosítása "
                                      + "folyamatban..."
                    );
                }

                for (Future<ParallelRecognitionResult>
                        recognitionFuture
                        : recognitionFutures) {
                    try {
                        ParallelRecognitionResult
                                parallelResult =
                                recognitionFuture.get();

                        parallelResults.set(
                                parallelResult.displayedIndex - 1,
                                parallelResult
                        );

                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();

                        throw new Exception(
                                "A párhuzamos gyógyszerfelismerés "
                                        + "megszakadt.",
                                interrupted
                        );

                    } catch (
                            java.util.concurrent.ExecutionException
                                    executionError
                    ) {
                        Throwable cause =
                                executionError.getCause();

                        throw new Exception(
                                "A párhuzamos gyógyszerfelismerés "
                                        + "váratlan hibával leállt: "
                                        + (cause == null
                                        ? executionError.getMessage()
                                        : cause.getMessage()),
                                cause == null
                                        ? executionError
                                        : cause
                        );
                    }
                }

                if (firstRecognitionSubmittedAt >= 0L) {
                    parallelRecognitionWallMs =
                            SystemClock.elapsedRealtime()
                                    - firstRecognitionSubmittedAt;
                }

                int successfullyProcessedCount = 0;
                int acceptedCount = 0;
                MatchResult firstAcceptedResult = null;

                StringBuilder combinedTopKReport = null;

                if (medicineCount > 1) {
                    combinedTopKReport = new StringBuilder();

                    combinedTopKReport.append(
                            "MEDSCAN TÖBB GYÓGYSZER TOP-K EREDMÉNY\n"
                    );
                    combinedTopKReport.append(
                            "=====================================\n"
                    );
                    combinedTopKReport.append("Forrásfájl: ")
                            .append(
                                    evaluationFilename == null
                                            || evaluationFilename
                                            .trim()
                                            .isEmpty()
                                            ? "kamerás kép / nincs fájlcímke"
                                            : evaluationFilename
                            )
                            .append("\n");
                    combinedTopKReport.append("Gyógyszerek száma: ")
                            .append(medicineCount)
                            .append("\n\n");
                }

                for (int index = 0;
                     index < medicineCount;
                     index++) {

                    int displayedIndex = index + 1;

                    GeminiEmbeddingService.SegmentationResult
                            segmentation =
                            segmentations.get(index);

                    ParallelRecognitionResult
                            parallelResult =
                            parallelResults.get(index);

                    if (parallelResult == null) {
                        parallelResult =
                                ParallelRecognitionResult.failure(
                                        displayedIndex,
                                        segmentation,
                                        0L,
                                        0L,
                                        new Exception(
                                                "A felismerési feladat "
                                                        + "nem adott eredményt."
                                        )
                                );
                    }

                    embeddingMs +=
                            parallelResult.embeddingMs;

                    databaseMs +=
                            parallelResult.databaseMs;

                    if (parallelResult.isSuccessful()) {
                        MatchResult result =
                                parallelResult.matchResult;

                        results.add(result);
                        successfullyProcessedCount++;

                        if (result.found) {
                            acceptedCount++;

                            if (firstAcceptedResult == null) {
                                firstAcceptedResult = result;
                            }
                        }

                        JSONObject resultItem =
                                new JSONObject();

                        if (medicineCount > 1) {
                            resultItem.put(
                                    "index",
                                    displayedIndex
                            );
                        }

                        resultItem.put(
                                "found",
                                result.found
                        );
                        resultItem.put(
                                "name",
                                result.name
                        );
                        resultItem.put(
                                "side",
                                result.side
                        );
                        resultItem.put(
                                "score",
                                result.score
                        );
                        resultItem.put(
                                "minimum_score",
                                MINIMUM_MATCH_SCORE
                        );
                        resultItem.put(
                                "margin_check_enabled",
                                false
                        );
                        resultItem.put(
                                "embedding_dimension",
                                parallelResult.embeddingDimension
                        );
                        resultItem.put(
                                "gemini_label",
                                segmentation.getLabel()
                        );

                        resultItems.put(resultItem);

                        String expectedForThisResult =
                                medicineCount == 1
                                        ? expectedMedicineName
                                        : "";

                        if (medicineCount == 1) {
                            String topKReport =
                                    buildTopKResultText(
                                            result,
                                            expectedForThisResult,
                                            evaluationFilename
                                    );

                            saveDebugTextQuietly(
                                    debugSaver,
                                    "11_top_k_results.txt",
                                    topKReport,
                                    "text/plain"
                            );
                        } else {
                            combinedTopKReport.append(
                                    buildMultiMedicineTopKSection(
                                            displayedIndex,
                                            segmentation.getLabel(),
                                            result,
                                            null
                                    )
                            );
                        }

                        if (medicineCount == 1
                                && expectedForThisResult != null
                                && !expectedForThisResult.trim().isEmpty()) {
                            String accuracyReport =
                                    updateAndBuildTopKAccuracyReport(
                                            appContext,
                                            result,
                                            expectedForThisResult,
                                            evaluationFilename
                                    );

                            saveDebugTextQuietly(
                                    debugSaver,
                                    "12_top_k_accuracy.txt",
                                    accuracyReport,
                                    "text/plain"
                            );
                        }

                        if (medicineCount > 1) {
                            saveDebugTextQuietly(
                                    debugSaver,
                                    String.format(
                                            Locale.ROOT,
                                            "10_pill_%02d_result.json",
                                            displayedIndex
                                    ),
                                    resultItem.toString(2),
                                    "application/json"
                            );
                        }

                    } else {
                        Exception recognitionError =
                                parallelResult.error;

                        String detail =
                                recognitionError != null
                                        && recognitionError.getMessage()
                                        != null
                                        ? recognitionError.getMessage()
                                        : "Ismeretlen felismerési hiba.";

                        MatchResult failedResult =
                                new MatchResult(
                                        false,
                                        "A(z) "
                                                + displayedIndex
                                                + ". gyógyszer "
                                                + "feldolgozása sikertelen.",
                                        "nincs",
                                        -1.0
                                );

                        results.add(failedResult);

                        JSONObject errorItem =
                                new JSONObject();

                        if (medicineCount > 1) {
                            errorItem.put(
                                    "index",
                                    displayedIndex
                            );
                        }

                        errorItem.put(
                                "found",
                                false
                        );
                        errorItem.put(
                                "gemini_label",
                                segmentation.getLabel()
                        );
                        errorItem.put(
                                "error",
                                detail
                        );

                        resultItems.put(errorItem);

                        if (medicineCount > 1) {
                            combinedTopKReport.append(
                                    buildMultiMedicineTopKSection(
                                            displayedIndex,
                                            segmentation.getLabel(),
                                            null,
                                            detail
                                    )
                            );
                        }

                        if (medicineCount > 1) {
                            saveDebugTextQuietly(
                                    debugSaver,
                                    String.format(
                                            Locale.ROOT,
                                            "10_pill_%02d_result.json",
                                            displayedIndex
                                    ),
                                    errorItem.toString(2),
                                    "application/json"
                            );
                        }
                    }
                }

                if (medicineCount > 1
                        && combinedTopKReport != null) {
                    saveDebugTextQuietly(
                            debugSaver,
                            "11_top_k_results.txt",
                            combinedTopKReport.toString(),
                            "text/plain"
                    );
                }

                if (successfullyProcessedCount == 0) {
                    throw new Exception(
                            "Egyik megtalált gyógyszer feldolgozása "
                                    + "sem sikerült."
                    );
                }

                long totalMs =
                        SystemClock.elapsedRealtime()
                                - totalStart;

                JSONObject runtimeJson =
                        new JSONObject();

                runtimeJson.put(
                        "session_id",
                        debugSaver.getSessionId()
                );
                runtimeJson.put(
                        "segmentation_model",
                        firstSegmentation.getModel()
                );
                runtimeJson.put(
                        "maximum_camera_height_cm",
                        MAX_CAMERA_HEIGHT_CM
                );

                if (!Double.isNaN(capturedCameraHeightCm)) {
                    runtimeJson.put(
                            "captured_camera_height_cm",
                            roundSeconds(capturedCameraHeightCm)
                    );
                } else {
                    runtimeJson.put(
                            "captured_camera_height_cm",
                            JSONObject.NULL
                    );
                }

                runtimeJson.put(
                        "aruco_seconds",
                        millisecondsToSeconds(arucoMs)
                );
                runtimeJson.put(
                        "search_crop_seconds",
                        millisecondsToSeconds(cropMs)
                );
                runtimeJson.put(
                        "segmentation_seconds",
                        millisecondsToSeconds(segmentationMs)
                );
                runtimeJson.put(
                        "segmentation_api_seconds",
                        roundSeconds(
                                firstSegmentation
                                        .getRuntimeSeconds()
                        )
                );
                runtimeJson.put(
                        "mask_and_overlay_seconds",
                        millisecondsToSeconds(maskMs)
                );
                runtimeJson.put(
                        "embedding_preparation_seconds",
                        millisecondsToSeconds(
                                embeddingPreparationMs
                        )
                );
                runtimeJson.put(
                        "embedding_api_seconds",
                        millisecondsToSeconds(embeddingMs)
                );
                runtimeJson.put(
                        "database_search_seconds",
                        millisecondsToSeconds(databaseMs)
                );
                runtimeJson.put(
                        "parallel_recognition_enabled",
                        medicineCount > 1
                );
                runtimeJson.put(
                        "parallel_recognition_threads",
                        PARALLEL_RECOGNITION_THREADS
                );
                runtimeJson.put(
                        "parallel_recognition_wall_seconds",
                        millisecondsToSeconds(
                                parallelRecognitionWallMs
                        )
                );
                runtimeJson.put(
                        "total_seconds",
                        millisecondsToSeconds(totalMs)
                );

                if (medicineCount > 1) {
                    runtimeJson.put(
                            "medicine_count",
                            medicineCount
                    );
                    runtimeJson.put(
                            "accepted_count",
                            acceptedCount
                    );
                }

                saveDebugTextQuietly(
                        debugSaver,
                        "09_runtime.json",
                        runtimeJson.toString(2),
                        "application/json"
                );

                JSONObject resultJson;

                if (medicineCount == 1
                        && resultItems.length() == 1) {
                    resultJson =
                            resultItems.getJSONObject(0);

                    resultJson.put(
                            "maximum_camera_height_cm",
                            MAX_CAMERA_HEIGHT_CM
                    );

                    if (!Double.isNaN(
                            capturedCameraHeightCm
                    )) {
                        resultJson.put(
                                "captured_camera_height_cm",
                                roundSeconds(
                                        capturedCameraHeightCm
                                )
                        );
                    } else {
                        resultJson.put(
                                "captured_camera_height_cm",
                                JSONObject.NULL
                        );
                    }
                } else {
                    resultJson =
                            new JSONObject();

                    resultJson.put(
                            "medicine_count",
                            medicineCount
                    );
                    resultJson.put(
                            "accepted_count",
                            acceptedCount
                    );
                    resultJson.put(
                            "minimum_score",
                            MINIMUM_MATCH_SCORE
                    );
                    resultJson.put(
                            "margin_check_enabled",
                            false
                    );
                    resultJson.put(
                            "embedding_dimension",
                            databaseDimension
                    );
                    resultJson.put(
                            "maximum_camera_height_cm",
                            MAX_CAMERA_HEIGHT_CM
                    );

                    if (!Double.isNaN(
                            capturedCameraHeightCm
                    )) {
                        resultJson.put(
                                "captured_camera_height_cm",
                                roundSeconds(
                                        capturedCameraHeightCm
                                )
                        );
                    } else {
                        resultJson.put(
                                "captured_camera_height_cm",
                                JSONObject.NULL
                        );
                    }

                    resultJson.put(
                            "results",
                            resultItems
                    );
                }

                saveDebugTextQuietly(
                        debugSaver,
                        "10_result.json",
                        resultJson.toString(2),
                        "application/json"
                );

                Bitmap finalPreview;

                if (medicineCount == 1
                        && singleResultPreviewBitmap != null) {
                    finalPreview =
                            singleResultPreviewBitmap;

                    singleResultPreviewBitmap = null;
                } else {
                    finalPreview =
                            combinedOverlayBitmap.copy(
                                    Bitmap.Config.ARGB_8888,
                                    false
                            );
                }

                final double displayedTotalSeconds =
                        millisecondsToSeconds(totalMs);

                String resultText;

                if (medicineCount == 1) {
                    MatchResult result =
                            results.get(0);

                    StringBuilder builder =
                            new StringBuilder();

                    if (result.found) {
                        builder.append(
                                "Legvalószínűbb találat:\n"
                        );
                        builder.append(
                                result.name
                        );
                        builder.append(
                                "\n\nAz eredmény képi hasonlóságon "
                                        + "alapul, és nem jelent biztos "
                                        + "gyógyszerazonosítást. Ellenőrizd "
                                        + "a csomagolást és a gyógyszer "
                                        + "jelöléseit."
                        );

                        if (BuildConfig.DEBUG) {
                            builder.append(
                                    "\n\n[DEBUG]"
                            );
                            builder.append(
                                    "\nAdatbázis-oldal: "
                            );
                            builder.append(
                                    result.side
                            );
                            builder.append(
                                    "\nHasonlóság: "
                            );
                            builder.append(
                                    String.format(
                                            Locale.getDefault(),
                                            "%.4f",
                                            result.score
                                    )
                            );
                            builder.append(
                                    "\nTeljes idő: "
                            );
                            builder.append(
                                    formatSeconds(
                                            displayedTotalSeconds
                                    )
                            );
                            builder.append(
                                    " másodperc"
                            );
                        }

                    } else {
                        builder.append(
                                "A hasonlóság nem érte el a szükséges "
                                        + "küszöböt."
                        );
                        builder.append(
                                "\nKészíts élesebb, közelebbi képet, "
                                        + "vagy válassz másik fotót."
                        );

                        if (BuildConfig.DEBUG) {
                            builder.append(
                                    "\n\n[DEBUG]\n"
                            );
                            builder.append(
                                    result.name
                            );
                            builder.append(
                                    "\nTeljes idő: "
                            );
                            builder.append(
                                    formatSeconds(
                                            displayedTotalSeconds
                                    )
                            );
                            builder.append(
                                    " másodperc"
                            );
                        }
                    }

                    resultText =
                            builder.toString();

                } else {
                    StringBuilder builder =
                            new StringBuilder();

                    for (int i = 0;
                         i < results.size();
                         i++) {
                        MatchResult result =
                                results.get(i);

                        if (i > 0) {
                            builder.append(
                                    "\n\n"
                            );
                        }

                        builder.append(
                                i + 1
                        );
                        builder.append(
                                ". gyógyszer\n"
                        );

                        if (result.found) {
                            builder.append(
                                    "Legvalószínűbb találat:\n"
                            );
                            builder.append(
                                    result.name
                            );

                            if (BuildConfig.DEBUG) {
                                builder.append(
                                        "\n[DEBUG] oldal="
                                );
                                builder.append(
                                        result.side
                                );
                                builder.append(
                                        ", hasonlóság="
                                );
                                builder.append(
                                        String.format(
                                                Locale.getDefault(),
                                                "%.4f",
                                                result.score
                                        )
                                );
                            }

                        } else {
                            builder.append(
                                    "A hasonlóság nem érte el a szükséges "
                                            + "küszöböt."
                            );

                            if (BuildConfig.DEBUG) {
                                builder.append(
                                        "\n[DEBUG] "
                                );
                                builder.append(
                                        result.name
                                );
                            }
                        }
                    }

                    if (acceptedCount < medicineCount) {
                        builder.append(
                                "\n\nAzokról a gyógyszerekről, amelyeknél "
                                        + "a hasonlóság nem érte el a "
                                        + "szükséges küszöböt, készíts "
                                        + "külön, közelebbi és élesebb képet."
                        );
                    }

                    if (BuildConfig.DEBUG) {
                        builder.append(
                                "\n\n[DEBUG] Teljes idő: "
                        );
                        builder.append(
                                formatSeconds(
                                        displayedTotalSeconds
                                )
                        );
                        builder.append(
                                " másodperc"
                        );
                    }

                    resultText =
                            builder.toString();
                }

                handler.post(() -> {
                    if (!isAdded()
                            || ivSelectedPill == null
                            || tvResultName == null) {
                        return;
                    }

                    ivSelectedPill.setImageBitmap(
                            finalPreview
                    );
                    tvResultName.setText(
                            resultText
                    );
                });

                Log.d(
                        "PillRuntime",
                        "Gyógyszerek="
                                + medicineCount
                                + "; elfogadott="
                                + acceptedCount
                                + "; ArUco="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        arucoMs
                                )
                        )
                                + " s; crop="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        cropMs
                                )
                        )
                                + " s; segmentation="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        segmentationMs
                                )
                        )
                                + " s; mask="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        maskMs
                                )
                        )
                                + " s; embeddingPrepare="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        embeddingPreparationMs
                                )
                        )
                                + " s; embedding="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        embeddingMs
                                )
                        )
                                + " s; dbWork="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        databaseMs
                                )
                        )
                                + " s; parallelWall="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        parallelRecognitionWallMs
                                )
                        )
                                + " s; total="
                                + formatSeconds(
                                millisecondsToSeconds(
                                        totalMs
                                )
                        )
                                + " s"
                );

                if (firstAcceptedResult != null) {
                    saveMedicineHistoryEntry(
                            appContext,
                            originalBeforeFilters,
                            firstAcceptedResult,
                            savedOriginalImageUri
                    );
                }

            } catch (Exception e) {
                Log.e(
                        "PillScan",
                        "Szegmentálási/felismerési hiba",
                        e
                );

                String technicalErrorMessage =
                        e.getMessage() != null
                                ? e.getMessage()
                                : e.getClass().getSimpleName();

                String userErrorMessage =
                        createUserFriendlyProcessingError(e);

                JSONObject errorJson =
                        new JSONObject();

                try {
                    errorJson.put(
                            "error",
                            technicalErrorMessage
                    );
                    errorJson.put(
                            "user_message",
                            userErrorMessage
                    );
                    errorJson.put(
                            "total_seconds",
                            millisecondsToSeconds(
                                    SystemClock.elapsedRealtime()
                                            - totalStart
                            )
                    );
                } catch (Exception ignored) {
                }

                saveDebugTextQuietly(
                        debugSaver,
                        "error.json",
                        errorJson.toString(),
                        "application/json"
                );

                handler.post(() ->
                        showError(
                                userErrorMessage,
                                e
                        )
                );

            } finally {
                if (inputMat != null) {
                    inputMat.release();
                }

                if (processedMat != null) {
                    processedMat.release();
                }

                if (correctedBitmap != null
                        && !correctedBitmap.isRecycled()) {
                    correctedBitmap.recycle();
                }

                if (searchAreaBitmap != null
                        && !searchAreaBitmap.isRecycled()) {
                    searchAreaBitmap.recycle();
                }

                if (combinedOverlayBitmap != null
                        && !combinedOverlayBitmap.isRecycled()) {
                    combinedOverlayBitmap.recycle();
                }

                if (singleResultPreviewBitmap != null
                        && !singleResultPreviewBitmap.isRecycled()) {
                    singleResultPreviewBitmap.recycle();
                }

                if (bitmap != null
                        && !bitmap.isRecycled()) {
                    bitmap.recycle();
                }

                if (originalBeforeFilters != null
                        && originalBeforeFilters != bitmap
                        && !originalBeforeFilters.isRecycled()) {
                    originalBeforeFilters.recycle();
                }

                isProcessing = false;
                hasEditablePillPhoto = false;
                setProcessingControlsEnabled(true);

                handler.post(() -> {
                    if (!isAdded()) {
                        return;
                    }

                    if (btnOpenCamera != null) {
                        btnOpenCamera.setText(
                                "Új kép készítése"
                        );
                    }
                });
            }
        });
    }

    private void saveMedicineHistoryEntry(
            Context appContext,
            Bitmap originalBeforeAruco,
            MatchResult result,
            Uri savedOriginalImageUri
    ) {
        if (appContext == null
                || result == null
                || !result.found) {
            Log.w(
                    "MedicineHistory",
                    "Nincs menthető elfogadott gyógyszertalálat."
            );
            return;
        }

        FirebaseUser currentUser =
                FirebaseAuth.getInstance().getCurrentUser();

        if (currentUser == null) {
            Log.w(
                    "MedicineHistory",
                    "Nincs bejelentkezett felhasználó, "
                            + "a gyógyszerelőzmény nem menthető."
            );
            return;
        }

        Uri historyImageUri =
                savedOriginalImageUri;

        File fallbackHistoryFile = null;

        if (historyImageUri == null) {
            if (originalBeforeAruco == null
                    || originalBeforeAruco.isRecycled()) {
                Log.e(
                        "MedicineHistory",
                        "Sem mentett kép-URI, sem használható eredeti "
                                + "Bitmap nem áll rendelkezésre."
                );
                return;
            }

            File historyDirectory =
                    new File(
                            appContext.getFilesDir(),
                            "history_images"
                    );

            if (!historyDirectory.exists()
                    && !historyDirectory.mkdirs()) {
                Log.e(
                        "MedicineHistory",
                        "Az előzményképek tartalékmappája "
                                + "nem hozható létre."
                );
                return;
            }

            long fallbackTimestamp =
                    System.currentTimeMillis();

            fallbackHistoryFile =
                    new File(
                            historyDirectory,
                            "medicine_original_"
                                    + fallbackTimestamp
                                    + ".jpg"
                    );

            try (FileOutputStream outputStream =
                         new FileOutputStream(
                                 fallbackHistoryFile
                         )) {
                boolean saved =
                        originalBeforeAruco.compress(
                                Bitmap.CompressFormat.JPEG,
                                95,
                                outputStream
                        );

                if (!saved) {
                    throw new IOException(
                            "Az eredeti gyógyszerfotó tartalék "
                                    + "JPEG-mentése sikertelen."
                    );
                }

                outputStream.flush();
                historyImageUri =
                        Uri.fromFile(
                                fallbackHistoryFile
                        );

            } catch (Exception saveError) {
                Log.e(
                        "MedicineHistory",
                        "Az eredeti gyógyszerfotó tartalék "
                                + "mentése sikertelen.",
                        saveError
                );

                if (fallbackHistoryFile.exists()) {

                    fallbackHistoryFile.delete();
                }

                return;
            }
        }

        final Uri finalHistoryImageUri =
                historyImageUri;

        final File finalFallbackHistoryFile =
                fallbackHistoryFile;

        long currentTimeMillis =
                System.currentTimeMillis();

        String formattedDate =
                new SimpleDateFormat(
                        "yyyy. MM. dd. HH:mm:ss",
                        Locale.getDefault()
                ).format(new Date(currentTimeMillis));

        Recept medicineHistory =
                new Recept(
                        result.name,
                        1,
                        "Gyógyszerszkennelés",
                        formattedDate,
                        null,
                        "",
                        finalHistoryImageUri.toString(),
                        currentUser.getUid()
                );

        medicineHistory.setHistoryType(
                Recept.TYPE_MEDICINE
        );

        medicineHistory.setSimilarity(
                result.score
        );

        medicineHistory.setRecognitionSide(
                result.side
        );

        FirebaseFirestore.getInstance()
                .collection("receptek")
                .add(medicineHistory)
                .addOnSuccessListener(documentReference ->
                        Log.d(
                                "MedicineHistory",
                                "Gyógyszerelőzmény mentve. "
                                        + "documentId="
                                        + documentReference.getId()
                                        + "; imageUri="
                                        + finalHistoryImageUri
                        )
                )
                .addOnFailureListener(saveError -> {
                    Log.e(
                            "MedicineHistory",
                            "A gyógyszerelőzmény Firestore-mentése "
                                    + "sikertelen.",
                            saveError
                    );

                    if (finalFallbackHistoryFile != null
                            && finalFallbackHistoryFile.exists()
                            && !finalFallbackHistoryFile.delete()) {
                        Log.w(
                                "MedicineHistory",
                                "A sikertelen rekord tartalék helyi "
                                        + "képe nem törölhető."
                        );
                    }
                });
    }

    private String getDisplayNameFromUri(Uri uri) {
        if (uri == null || !isAdded()) {
            return "";
        }

        Cursor cursor = null;

        try {
            cursor = requireContext()
                    .getContentResolver()
                    .query(
                            uri,
                            new String[]{
                                    MediaStore.MediaColumns.DISPLAY_NAME
                            },
                            null,
                            null,
                            null
                    );

            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(
                        MediaStore.MediaColumns.DISPLAY_NAME
                );

                if (index >= 0) {
                    String value = cursor.getString(index);
                    return value == null ? "" : value;
                }
            }
        } catch (Exception error) {
            Log.w(
                    "TopKAccuracy",
                    "A galériakép fájlneve nem olvasható.",
                    error
            );
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }

        return "";
    }

    private String resolveExpectedMedicineNameFromFilename(
            Context context,
            String filename
    ) {
        if (context == null
                || filename == null
                || filename.trim().isEmpty()) {
            return "";
        }

        String normalizedFilename =
                normalizeTopKName(filename);

        if (normalizedFilename.isEmpty()) {
            return "";
        }

        DatabaseManager db = null;
        Cursor cursor = null;
        String bestMatch = "";
        int bestNormalizedLength = 0;

        try {
            db = new DatabaseManager(context);
            cursor = db.getAllMedications();

            if (cursor == null) {
                return "";
            }

            while (cursor.moveToNext()) {
                String medicineName = cursor.getString(0);
                String normalizedMedicine =
                        normalizeTopKName(medicineName);

                if (!normalizedMedicine.isEmpty()
                        && normalizedFilename.contains(
                        normalizedMedicine
                )
                        && normalizedMedicine.length()
                        > bestNormalizedLength) {
                    bestMatch = medicineName;
                    bestNormalizedLength =
                            normalizedMedicine.length();
                }
            }
        } catch (Exception error) {
            Log.w(
                    "TopKAccuracy",
                    "A helyes címke nem határozható meg a fájlnévből.",
                    error
            );
        } finally {
            if (cursor != null) {
                cursor.close();
            }

            if (db != null) {
                db.close();
            }
        }

        Log.d(
                "TopKAccuracy",
                "Tesztfájl="
                        + filename
                        + "; helyes címke="
                        + (bestMatch.isEmpty()
                        ? "nincs"
                        : bestMatch)
        );

        return bestMatch;
    }

    private String normalizeTopKName(String value) {
        if (value == null) {
            return "";
        }

        String normalized = Normalizer.normalize(
                value,
                Normalizer.Form.NFD
        );

        return normalized
                .replaceAll("\\p{M}+", "")
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]", "");
    }

    private boolean isExpectedMedicineInTopK(
            MatchResult result,
            String expectedMedicineName,
            int k
    ) {
        if (result == null
                || expectedMedicineName == null
                || expectedMedicineName.trim().isEmpty()
                || result.topCandidates == null
                || result.topCandidates.isEmpty()) {
            return false;
        }

        String expectedNormalized =
                normalizeTopKName(expectedMedicineName);

        int limit = Math.min(
                Math.max(0, k),
                result.topCandidates.size()
        );

        for (int index = 0; index < limit; index++) {
            RankedMedicineCandidate candidate =
                    result.topCandidates.get(index);

            if (normalizeTopKName(candidate.name)
                    .equals(expectedNormalized)) {
                return true;
            }
        }

        return false;
    }

    private String buildMultiMedicineTopKSection(
            int displayedIndex,
            String geminiLabel,
            MatchResult result,
            String errorMessage
    ) {
        StringBuilder builder = new StringBuilder();

        builder.append("GYÓGYSZER ")
                .append(displayedIndex)
                .append("\n");
        builder.append("------------\n");
        builder.append("Gemini címke: ")
                .append(
                        geminiLabel == null
                                || geminiLabel.trim().isEmpty()
                                ? "medicine"
                                : geminiLabel
                )
                .append("\n");

        if (errorMessage != null
                && !errorMessage.trim().isEmpty()) {
            builder.append("Feldolgozási hiba: ")
                    .append(errorMessage)
                    .append("\n\n");
            return builder.toString();
        }

        if (result == null
                || result.topCandidates == null
                || result.topCandidates.isEmpty()) {
            builder.append(
                    "Nincs rangsorolható adatbázis-jelölt.\n\n"
            );
            return builder.toString();
        }

        builder.append("Rangsor:\n");

        for (int index = 0;
             index < result.topCandidates.size();
             index++) {
            RankedMedicineCandidate candidate =
                    result.topCandidates.get(index);

            builder.append(index + 1)
                    .append(". ")
                    .append(candidate.name)
                    .append(" | ")
                    .append(candidate.side)
                    .append(" | score=")
                    .append(String.format(
                            Locale.ROOT,
                            "%.6f",
                            candidate.score
                    ))
                    .append("\n");
        }

        builder.append("\nTop-1 jelöltek: ")
                .append(joinTopCandidateNames(result, 1))
                .append("\n");
        builder.append("Top-3 jelöltek: ")
                .append(joinTopCandidateNames(result, 3))
                .append("\n");
        builder.append("Top-5 jelöltek: ")
                .append(joinTopCandidateNames(result, 5))
                .append("\n\n");

        return builder.toString();
    }

    private String buildTopKResultText(
            MatchResult result,
            String expectedMedicineName,
            String sourceFilename
    ) {
        StringBuilder builder = new StringBuilder();

        builder.append("MEDSCAN TOP-K EREDMÉNY\n");
        builder.append("======================\n");
        builder.append("Forrásfájl: ")
                .append(sourceFilename == null
                        || sourceFilename.trim().isEmpty()
                        ? "kamerás kép / nincs fájlcímke"
                        : sourceFilename)
                .append("\n");

        boolean hasExpected =
                expectedMedicineName != null
                        && !expectedMedicineName.trim().isEmpty();

        builder.append("Helyes címke: ")
                .append(hasExpected
                        ? expectedMedicineName
                        : "nincs megadva")
                .append("\n\n");

        if (result == null
                || result.topCandidates == null
                || result.topCandidates.isEmpty()) {
            builder.append("Nincs rangsorolható adatbázis-jelölt.\n");
            return builder.toString();
        }

        builder.append("Rangsor:\n");

        for (int index = 0;
             index < result.topCandidates.size();
             index++) {
            RankedMedicineCandidate candidate =
                    result.topCandidates.get(index);

            builder.append(index + 1)
                    .append(". ")
                    .append(candidate.name)
                    .append(" | ")
                    .append(candidate.side)
                    .append(" | score=")
                    .append(String.format(
                            Locale.ROOT,
                            "%.6f",
                            candidate.score
                    ))
                    .append("\n");
        }

        builder.append("\nTop-1 jelöltek: ")
                .append(joinTopCandidateNames(result, 1))
                .append("\n");
        builder.append("Top-3 jelöltek: ")
                .append(joinTopCandidateNames(result, 3))
                .append("\n");
        builder.append("Top-5 jelöltek: ")
                .append(joinTopCandidateNames(result, 5))
                .append("\n");

        if (hasExpected) {
            builder.append("\nTop-1 találat: ")
                    .append(isExpectedMedicineInTopK(
                            result,
                            expectedMedicineName,
                            1
                    ) ? "IGEN" : "NEM")
                    .append("\n");
            builder.append("Top-3 találat: ")
                    .append(isExpectedMedicineInTopK(
                            result,
                            expectedMedicineName,
                            3
                    ) ? "IGEN" : "NEM")
                    .append("\n");
            builder.append("Top-5 találat: ")
                    .append(isExpectedMedicineInTopK(
                            result,
                            expectedMedicineName,
                            5
                    ) ? "IGEN" : "NEM")
                    .append("\n");
        } else {
            builder.append(
                    "\nAccuracy nem számolható, mert a tesztkép "
                            + "fájlnevéből nem állapítható meg a helyes "
                            + "gyógyszernév.\n"
            );
        }

        return builder.toString();
    }

    private String joinTopCandidateNames(
            MatchResult result,
            int k
    ) {
        if (result == null
                || result.topCandidates == null
                || result.topCandidates.isEmpty()) {
            return "nincs";
        }

        StringBuilder builder = new StringBuilder();
        int limit = Math.min(k, result.topCandidates.size());

        for (int index = 0; index < limit; index++) {
            if (index > 0) {
                builder.append(", ");
            }

            builder.append(
                    result.topCandidates.get(index).name
            );
        }

        return builder.toString();
    }

    private String updateAndBuildTopKAccuracyReport(
            Context context,
            MatchResult result,
            String expectedMedicineName,
            String sourceFilename
    ) {
        boolean top1Hit = isExpectedMedicineInTopK(
                result,
                expectedMedicineName,
                1
        );
        boolean top3Hit = isExpectedMedicineInTopK(
                result,
                expectedMedicineName,
                3
        );
        boolean top5Hit = isExpectedMedicineInTopK(
                result,
                expectedMedicineName,
                5
        );

        android.content.SharedPreferences preferences =
                context.getSharedPreferences(
                        TOP_K_PREFERENCES,
                        Context.MODE_PRIVATE
                );

        int total = preferences.getInt("total", 0) + 1;
        int top1Correct =
                preferences.getInt("top1_correct", 0)
                        + (top1Hit ? 1 : 0);
        int top3Correct =
                preferences.getInt("top3_correct", 0)
                        + (top3Hit ? 1 : 0);
        int top5Correct =
                preferences.getInt("top5_correct", 0)
                        + (top5Hit ? 1 : 0);

        preferences.edit()
                .putInt("total", total)
                .putInt("top1_correct", top1Correct)
                .putInt("top3_correct", top3Correct)
                .putInt("top5_correct", top5Correct)
                .apply();

        double top1Accuracy =
                top1Correct * 100.0 / total;
        double top3Accuracy =
                top3Correct * 100.0 / total;
        double top5Accuracy =
                top5Correct * 100.0 / total;

        StringBuilder builder = new StringBuilder();
        builder.append("MEDSCAN TOP-K ACCURACY\n");
        builder.append("=======================\n");
        builder.append("Kiértékelt minták száma: ")
                .append(total)
                .append("\n");
        builder.append("Top-1 accuracy: ")
                .append(String.format(
                        Locale.ROOT,
                        "%.2f%% (%d/%d)",
                        top1Accuracy,
                        top1Correct,
                        total
                ))
                .append("\n");
        builder.append("Top-3 accuracy: ")
                .append(String.format(
                        Locale.ROOT,
                        "%.2f%% (%d/%d)",
                        top3Accuracy,
                        top3Correct,
                        total
                ))
                .append("\n");
        builder.append("Top-5 accuracy: ")
                .append(String.format(
                        Locale.ROOT,
                        "%.2f%% (%d/%d)",
                        top5Accuracy,
                        top5Correct,
                        total
                ))
                .append("\n\n");
        builder.append("Legutóbbi teszt:\n");
        builder.append("Fájl: ")
                .append(sourceFilename)
                .append("\n");
        builder.append("Helyes címke: ")
                .append(expectedMedicineName)
                .append("\n");
        builder.append("Top-1: ")
                .append(top1Hit ? "találat" : "hiba")
                .append("\n");
        builder.append("Top-3: ")
                .append(top3Hit ? "találat" : "hiba")
                .append("\n");
        builder.append("Top-5: ")
                .append(top5Hit ? "találat" : "hiba")
                .append("\n");

        return builder.toString();
    }

    private MatchResult findBestMatch(
            Context context,
            float[] queryVector
    ) {
        DatabaseManager db = null;
        Cursor cursor = null;

        List<RankedMedicineCandidate> rankedCandidates =
                new ArrayList<>();

        try {
            db = new DatabaseManager(context);
            cursor = db.getAllMedications();

            if (cursor == null || cursor.getCount() == 0) {
                return new MatchResult(
                        false,
                        "Üres adatbázis",
                        "nincs",
                        -1.0,
                        rankedCandidates
                );
            }

            while (cursor.moveToNext()) {
                String name = cursor.getString(0);
                String leftStr = cursor.getString(1);
                String rightStr = cursor.getString(2);

                float[] leftVector =
                        safeVectorFromString(leftStr);

                float[] rightVector =
                        safeVectorFromString(rightStr);

                double leftScore = cosineSimilarity(
                        queryVector,
                        leftVector
                );

                double rightScore = cosineSimilarity(
                        queryVector,
                        rightVector
                );

                double score;
                String side;

                if (leftScore >= rightScore) {
                    score = leftScore;
                    side = "bal oldal";
                } else {
                    score = rightScore;
                    side = "jobb oldal";
                }

                if (Double.isNaN(score) || score < 0.0) {
                    continue;
                }

                rankedCandidates.add(
                        new RankedMedicineCandidate(
                                name,
                                side,
                                score
                        )
                );
            }

        } catch (Exception error) {
            return new MatchResult(
                    false,
                    "Adatbázis hiba: " + error.getMessage(),
                    "nincs",
                    -1.0,
                    rankedCandidates
            );

        } finally {
            if (cursor != null) {
                cursor.close();
            }

            if (db != null) {
                db.close();
            }
        }

        rankedCandidates.sort(
                (first, second) ->
                        Double.compare(
                                second.score,
                                first.score
                        )
        );

        if (rankedCandidates.size()
                > TOP_K_MAX_CANDIDATES) {
            rankedCandidates =
                    new ArrayList<>(
                            rankedCandidates.subList(
                                    0,
                                    TOP_K_MAX_CANDIDATES
                            )
                    );
        }

        if (rankedCandidates.isEmpty()) {
            return new MatchResult(
                    false,
                    "Nincs összehasonlítható embedding. Ellenőrizd, hogy a "
                            + "telefonos és adatbázis-vektorok egyaránt "
                            + "512 dimenziósak.",
                    "nincs",
                    -1.0,
                    rankedCandidates
            );
        }

        RankedMedicineCandidate best =
                rankedCandidates.get(0);

        if (best.score < MINIMUM_MATCH_SCORE) {
            return new MatchResult(
                    false,
                    "Legjobb jelölt: "
                            + best.name
                            + " ("
                            + best.side
                            + "), hasonlóság: "
                            + String.format(
                            Locale.getDefault(),
                            "%.4f",
                            best.score
                    )
                            + ", szükséges minimum: "
                            + String.format(
                            Locale.getDefault(),
                            "%.4f",
                            MINIMUM_MATCH_SCORE
                    ),
                    best.side,
                    best.score,
                    rankedCandidates
            );
        }

        return new MatchResult(
                true,
                best.name,
                best.side,
                best.score,
                rankedCandidates
        );
    }

    private float[] safeVectorFromString(String value) {

        if (value == null || value.trim().isEmpty()) {

            return new float[0];

        }

        try {

            return DatabaseManager.stringToFloatArray(value);

        } catch (Exception e) {

            return new float[0];

        }

    }

    private double cosineSimilarity(float[] a, float[] b) {

        if (a == null || b == null || a.length == 0 || b.length == 0) {

            return -1.0;

        }

        if (a.length != b.length) {

            return -1.0;

        }

        double dot = 0.0;

        double normA = 0.0;

        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {

            dot += a[i] * b[i];

            normA += a[i] * a[i];

            normB += b[i] * b[i];

        }

        if (normA == 0.0 || normB == 0.0) {

            return -1.0;

        }

        return dot / (Math.sqrt(normA) * Math.sqrt(normB));

    }

    private void showSavedFilename(
            String filename
    ) {
        if (filename == null
                || filename.trim().isEmpty()) {
            return;
        }

        String cleanFilename =
                new File(filename).getName();

        handler.post(() -> {
            if (!isAdded()
                    || tvResultName == null) {
                return;
            }

            tvResultName.setText(
                    BuildConfig.DEBUG
                            ? "A fénykép mentése sikerült.\n"
                              + "[DEBUG] Fájl: "
                              + cleanFilename
                            : "A fénykép mentése sikerült. "
                              + "A felismerés folytatódik."
            );
        });
    }

    private void showProcessingStatus(
            String status
    ) {
        handler.post(() -> {
            if (!isAdded()
                    || tvResultName == null) {
                return;
            }

            tvResultName.setText(status);
        });
    }

    private String createUserFriendlyProcessingError(
            Throwable error
    ) {
        String serviceMessage =
                GeminiEmbeddingService
                        .getUserFriendlyErrorMessage(
                                error
                        );

        if (serviceMessage != null
                && !serviceMessage.trim().isEmpty()) {
            return serviceMessage;
        }

        String technicalMessage =
                error == null
                        || error.getMessage() == null
                        ? ""
                        : error.getMessage();

        String normalized =
                technicalMessage.toLowerCase(
                        Locale.ROOT
                );

        if (error instanceof OutOfMemoryError
                || normalized.contains("outofmemory")
                || normalized.contains("not enough memory")
                || normalized.contains("nincs elég memória")) {
            return "A kép túl nagy a készülék számára. "
                    + "Válassz kisebb felbontású képet, "
                    + "vagy készíts új fényképet.";
        }

        if (normalized.contains("adatbázis")
                || normalized.contains("database")
                || normalized.contains("sqlite")
                || normalized.contains("cursor")
                || normalized.contains("no such table")) {
            return "A gyógyszeradatbázist nem sikerült megnyitni. "
                    + "Indítsd újra az alkalmazást, majd próbáld újra.";
        }

        if (error instanceof IndexOutOfBoundsException
                || error instanceof ArrayIndexOutOfBoundsException
                || normalized.contains("index")
                || normalized.contains("out of range")
                || normalized.contains("out of bounds")
                || normalized.contains("[0..0)")) {
            return "A képen nem volt elegendő használható részlet. "
                    + "Mutasd a teljes lapot és mind a négy "
                    + "sarokjelölést, majd készíts új fényképet.";
        }

        if (normalized.contains("aruco")
                || normalized.contains("marker")) {
            return "Nem sikerült beolvasni a lap mind a négy "
                    + "sarokjelölését. A teljes lap legyen a képen, "
                    + "és egyik jelölést se takard le.";
        }

        if (normalized.contains("segment")
                || normalized.contains("mask")
                || normalized.contains("gyógyszerszegmens")
                || normalized.contains("nem talált feldolgozható")) {
            return "Nem sikerült megfelelően elkülöníteni a "
                    + "gyógyszert a háttértől. Tedd a gyógyszert "
                    + "a lap közepére, kerüld az erős árnyékot, "
                    + "majd készíts új képet.";
        }

        if (normalized.contains("embedding")
                || normalized.contains("vector")
                || normalized.contains("dimension")
                || normalized.contains("dimenzió")) {
            return "A gyógyszer összehasonlításához szükséges "
                    + "képi jellemzők nem készültek el. "
                    + "Próbáld újra egy élesebb képpel.";
        }

        if (normalized.contains("bitmap")
                || normalized.contains("képformátum")
                || normalized.contains("decode")
                || normalized.contains("jpeg")
                || normalized.contains("yuv")) {
            return "A kiválasztott képet nem sikerült beolvasni. "
                    + "Válassz másik képet, vagy készíts új fényképet.";
        }

        if (normalized.contains("camera")
                || normalized.contains("camerax")
                || normalized.contains("capture")) {
            return "A kamera nem tudta elkészíteni a fényképet. "
                    + "Tartsd stabilan a készüléket, majd próbáld újra.";
        }

        if (normalized.contains("json")
                || normalized.contains("response")
                || normalized.contains("válasz")) {
            return "Az online képelemző szolgáltatás hibás "
                    + "választ adott. Próbáld újra.";
        }

        return "A gyógyszer felismerése most nem sikerült. "
                + "Ellenőrizd, hogy a teljes lap, mind a négy "
                + "sarokjelölés és a gyógyszer jól látható legyen, "
                + "majd próbáld újra.";
    }

    private boolean isFullScreenCameraVisible() {
        return fullCameraContainer != null
                && fullCameraContainer.getVisibility()
                == View.VISIBLE;
    }

    private static final class LiveArucoMarker {
        final int id;
        final Point[] corners;

        LiveArucoMarker(
                int id,
                Point[] corners
        ) {
            this.id =
                    id;

            this.corners =
                    corners == null
                            ? new Point[0]
                            : corners.clone();
        }
    }

    private static final class LiveArucoDetectionResult {
        final List<Integer> detectedIds;
        final List<Integer> missingIds;
        final List<LiveArucoMarker> markers;
        final List<Point[]> rejectedCandidates;
        final boolean allRequiredFound;
        final String errorMessage;

        LiveArucoDetectionResult(
                List<Integer> detectedIds,
                List<Integer> missingIds,
                List<LiveArucoMarker> markers,
                List<Point[]> rejectedCandidates,
                boolean allRequiredFound
        ) {
            this(
                    detectedIds,
                    missingIds,
                    markers,
                    rejectedCandidates,
                    allRequiredFound,
                    ""
            );
        }

        LiveArucoDetectionResult(
                List<Integer> detectedIds,
                List<Integer> missingIds,
                List<LiveArucoMarker> markers,
                List<Point[]> rejectedCandidates,
                boolean allRequiredFound,
                String errorMessage
        ) {
            this.detectedIds =
                    new ArrayList<>(
                            detectedIds == null
                                    ? Collections.emptyList()
                                    : detectedIds
                    );

            this.missingIds =
                    new ArrayList<>(
                            missingIds == null
                                    ? Collections.emptyList()
                                    : missingIds
                    );

            this.markers =
                    new ArrayList<>(
                            markers == null
                                    ? Collections.emptyList()
                                    : markers
                    );

            this.rejectedCandidates =
                    new ArrayList<>();

            if (rejectedCandidates != null) {
                for (Point[] candidate
                        : rejectedCandidates) {
                    this.rejectedCandidates.add(
                            candidate == null
                                    ? new Point[0]
                                    : candidate.clone()
                    );
                }
            }

            this.allRequiredFound =
                    allRequiredFound;

            this.errorMessage =
                    errorMessage == null
                            ? ""
                            : errorMessage;
        }

        static LiveArucoDetectionResult empty() {
            List<Integer> missing =
                    new ArrayList<>();

            missing.add(0);
            missing.add(1);
            missing.add(2);
            missing.add(3);

            return new LiveArucoDetectionResult(
                    Collections.emptyList(),
                    missing,
                    Collections.emptyList(),
                    Collections.emptyList(),
                    false,
                    "Az élő kamerakép üres."
            );
        }

        static LiveArucoDetectionResult error(
                String message
        ) {
            List<Integer> missing =
                    new ArrayList<>();

            missing.add(0);
            missing.add(1);
            missing.add(2);
            missing.add(3);

            return new LiveArucoDetectionResult(
                    Collections.emptyList(),
                    missing,
                    Collections.emptyList(),
                    Collections.emptyList(),
                    false,
                    message
            );
        }
    }

    private static final class LiveArucoFrameHistory {
        final long elapsedMs;
        final int imageProxyWidth;
        final int imageProxyHeight;
        final int rotationDegrees;
        final int matWidth;
        final int matHeight;
        final int matChannels;
        final List<Integer> detectedIds;
        final List<Integer> missingIds;
        final int rejectedCandidates;
        final String errorMessage;

        LiveArucoFrameHistory(
                long elapsedMs,
                int imageProxyWidth,
                int imageProxyHeight,
                int rotationDegrees,
                int matWidth,
                int matHeight,
                int matChannels,
                List<Integer> detectedIds,
                List<Integer> missingIds,
                int rejectedCandidates,
                String errorMessage
        ) {
            this.elapsedMs =
                    elapsedMs;

            this.imageProxyWidth =
                    imageProxyWidth;

            this.imageProxyHeight =
                    imageProxyHeight;

            this.rotationDegrees =
                    rotationDegrees;

            this.matWidth =
                    matWidth;

            this.matHeight =
                    matHeight;

            this.matChannels =
                    matChannels;

            this.detectedIds =
                    new ArrayList<>(
                            detectedIds == null
                                    ? Collections.emptyList()
                                    : detectedIds
                    );

            this.missingIds =
                    new ArrayList<>(
                            missingIds == null
                                    ? Collections.emptyList()
                                    : missingIds
                    );

            this.rejectedCandidates =
                    rejectedCandidates;

            this.errorMessage =
                    errorMessage == null
                            ? ""
                            : errorMessage;
        }

        JSONObject toJson()
                throws Exception {
            JSONObject json =
                    new JSONObject();

            json.put(
                    "elapsed_ms",
                    elapsedMs
            );

            json.put(
                    "image_proxy_size",
                    imageProxyWidth
                            + "x"
                            + imageProxyHeight
            );

            json.put(
                    "rotation_degrees",
                    rotationDegrees
            );

            json.put(
                    "opencv_size",
                    matWidth
                            + "x"
                            + matHeight
            );

            json.put(
                    "opencv_channels",
                    matChannels
            );

            JSONArray detected =
                    new JSONArray();

            for (Integer id
                    : detectedIds) {
                detected.put(
                        id
                );
            }

            json.put(
                    "detected_ids",
                    detected
            );

            JSONArray missing =
                    new JSONArray();

            for (Integer id
                    : missingIds) {
                missing.put(
                        id
                );
            }

            json.put(
                    "missing_ids",
                    missing
            );

            json.put(
                    "rejected_candidates",
                    rejectedCandidates
            );

            if (!errorMessage.isEmpty()) {
                json.put(
                        "error",
                        errorMessage
                );
            }

            return json;
        }
    }

    private void showError(
            String userMessage
    ) {
        showError(
                userMessage,
                null
        );
    }

    private void showError(
            String userMessage,
            Throwable technicalError
    ) {
        if (tvResultName == null) {
            return;
        }

        String safeUserMessage =
                userMessage == null
                        || userMessage.trim().isEmpty()
                        ? "A művelet most nem sikerült. "
                          + "Kérlek, próbáld újra."
                        : userMessage.trim();

        if (BuildConfig.DEBUG
                && technicalError != null) {
            String debugDetails =
                    buildDebugErrorDetails(
                            technicalError
                    );

            if (!debugDetails.isEmpty()) {
                safeUserMessage +=
                        "\n\n[DEBUG]\n"
                                + debugDetails;
            }
        }

        tvResultName.setText(
                safeUserMessage
        );
    }

    private String buildDebugErrorDetails(
            Throwable error
    ) {
        if (error == null) {
            return "";
        }

        StringBuilder builder =
                new StringBuilder();

        Throwable current =
                error;

        int depth =
                0;

        while (current != null
                && depth < 4) {
            if (depth > 0) {
                builder.append(
                        "\nOkozta: "
                );
            }

            builder.append(
                    current.getClass()
                            .getSimpleName()
            );

            String message =
                    current.getMessage();

            if (message != null
                    && !message.trim().isEmpty()) {
                builder.append(
                        ": "
                );
                builder.append(
                        shortenDebugText(
                                message.trim(),
                                700
                        )
                );
            }

            current =
                    current.getCause();

            depth++;
        }

        return builder.toString();
    }

    private String shortenDebugText(
            String text,
            int maximumLength
    ) {
        if (text == null) {
            return "";
        }

        if (text.length() <= maximumLength) {
            return text;
        }

        return text.substring(
                0,
                maximumLength
        ) + "...";
    }

    private double millisecondsToSeconds(long milliseconds) {
        return roundSeconds(milliseconds / 1000.0);
    }

    private double roundSeconds(double seconds) {
        return Math.round(seconds * 1000.0) / 1000.0;
    }

    private String formatSeconds(double seconds) {
        return String.format(
                Locale.getDefault(),
                "%.3f",
                seconds
        );
    }

    private int clamp(int value, int min, int max) {

        return Math.max(min, Math.min(max, value));

    }

    private int dpToPx(int dp) {

        return Math.round(dp * getResources().getDisplayMetrics().density);

    }

    @Override

    public void onDestroyView() {

        super.onDestroyView();

        imageCapture = null;

        cameraBackCallback = null;

        ivSelectedPill = null;
        sliderContrast = null;
        btnFlash = null;

        viewFinder = null;

        tvResultName = null;

        tvCameraStatus = null;

        btnOpenCamera = null;

        btnOpenGallery = null;

        btnTakePhotoFull = null;

        btnCloseCamera = null;

        levelIndicator = null;

        fullCameraContainer = null;

        currentEvaluationFilename = "";

        synchronized (liveArucoDiagnosticLock) {
            liveArucoDiagnosticHistory.clear();
            liveArucoDiagnosticStartedElapsed = 0L;
            liveArucoDiagnosticSaved = false;
            liveArucoDiagnosticSaving = false;
            liveArucoEverFoundAll = false;
            liveArucoDiagnosticSessionName = "";
        }

        if (processedPillBitmap != null
                && !processedPillBitmap.isRecycled()) {
            processedPillBitmap.recycle();
        }

        if (currentBitmap != null
                && !currentBitmap.isRecycled()) {
            currentBitmap.recycle();
        }

        processedPillBitmap = null;
        currentBitmap = null;
        hasEditablePillPhoto = false;

    }

    @Override

    public void onDestroy() {

        super.onDestroy();

        executor.shutdown();
        recognitionExecutor.shutdown();

    }

    private static final class ParallelRecognitionResult {

        final int displayedIndex;
        final GeminiEmbeddingService.SegmentationResult segmentation;
        final MatchResult matchResult;
        final int embeddingDimension;
        final long embeddingMs;
        final long databaseMs;
        final Exception error;

        private ParallelRecognitionResult(
                int displayedIndex,
                GeminiEmbeddingService.SegmentationResult segmentation,
                MatchResult matchResult,
                int embeddingDimension,
                long embeddingMs,
                long databaseMs,
                Exception error
        ) {
            this.displayedIndex = displayedIndex;
            this.segmentation = segmentation;
            this.matchResult = matchResult;
            this.embeddingDimension = embeddingDimension;
            this.embeddingMs = embeddingMs;
            this.databaseMs = databaseMs;
            this.error = error;
        }

        static ParallelRecognitionResult success(
                int displayedIndex,
                GeminiEmbeddingService.SegmentationResult segmentation,
                MatchResult matchResult,
                int embeddingDimension,
                long embeddingMs,
                long databaseMs
        ) {
            return new ParallelRecognitionResult(
                    displayedIndex,
                    segmentation,
                    matchResult,
                    embeddingDimension,
                    embeddingMs,
                    databaseMs,
                    null
            );
        }

        static ParallelRecognitionResult failure(
                int displayedIndex,
                GeminiEmbeddingService.SegmentationResult segmentation,
                long embeddingMs,
                long databaseMs,
                Exception error
        ) {
            return new ParallelRecognitionResult(
                    displayedIndex,
                    segmentation,
                    null,
                    0,
                    embeddingMs,
                    databaseMs,
                    error
            );
        }

        boolean isSuccessful() {
            return error == null
                    && matchResult != null;
        }
    }

    private static final class RankedMedicineCandidate {

        final String name;
        final String side;
        final double score;

        RankedMedicineCandidate(
                String name,
                String side,
                double score
        ) {
            this.name = name;
            this.side = side;
            this.score = score;
        }
    }

    private static class MatchResult {

        boolean found;
        String name;
        String side;
        double score;
        List<RankedMedicineCandidate> topCandidates;

        MatchResult(
                boolean found,
                String name,
                String side,
                double score
        ) {
            this(
                    found,
                    name,
                    side,
                    score,
                    new ArrayList<>()
            );
        }

        MatchResult(
                boolean found,
                String name,
                String side,
                double score,
                List<RankedMedicineCandidate> topCandidates
        ) {
            this.found = found;
            this.name = name;
            this.side = side;
            this.score = score;
            this.topCandidates =
                    topCandidates == null
                            ? new ArrayList<>()
                            : new ArrayList<>(topCandidates);
        }
    }

}