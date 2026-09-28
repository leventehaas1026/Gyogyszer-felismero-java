package com.example.kmd5a9_szakmaigyakorlat;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
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
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.ai.client.generativeai.GenerativeModel;
import com.google.ai.client.generativeai.java.GenerativeModelFutures;
import com.google.ai.client.generativeai.type.Content;
import com.google.ai.client.generativeai.type.GenerateContentResponse;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.firebase.firestore.FirebaseFirestore;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.Normalizer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class ScanFragment extends Fragment implements SensorEventListener {

    private static final float LEVEL_TOLERANCE_DEGREES = 15.0f;
    private static final float MIN_FACE_UP_Z = 7.0f;
    private static final float SENSOR_SMOOTHING_ALPHA = 0.18f;

    private MaterialButton scanBtn;
    private MaterialButton galleryBtn;
    private TextView resultTextView;
    private ImageView receptImageView;
    private SeekBar sliderContrast;
    private FloatingActionButton btnFlash;

    private PreviewView viewFinder;
    private TextView tvCameraStatus;
    private MaterialButton btnTakePhotoFull;
    private MaterialButton btnCloseCamera;
    private View levelIndicator;
    private View fullCameraContainer;

    private ActivityResultLauncher<String> requestPermissionLauncher;
    private ActivityResultLauncher<String> galleryLauncher;

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private boolean isPhoneLevel = false;

    private float filteredX = 0f;
    private float filteredY = 0f;
    private float filteredZ = 9.81f;
    private boolean hasSensorFilter = false;

    private ImageCapture imageCapture;
    private boolean cameraStarted = false;
    private boolean isTakingPhoto = false;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final java.util.concurrent.ExecutorService cameraExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();

    // Képfeldolgozáshoz szükséges változók
    private Bitmap baseBitmap = null;       // A kamerából kapott eredeti nyers kép
    private Bitmap processedBitmap = null;  // A csúszkák által módosított kész kép
    private int contrastProgress = 50;
    private boolean isFlashSimulated = false;
    private boolean hasPhotoBeenTaken = false; // Figyeli, hogy szerkesztő módban vagyunk-e

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        sensorManager = (SensorManager) requireActivity().getSystemService(Context.SENSOR_SERVICE);

        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        }

        /*
         * Receptkép kiválasztása a galériából.
         *
         * A kiválasztott kép ugyanabba a szerkesztési folyamatba kerül,
         * mint a kamerával készített receptfotó:
         * - EXIF szerinti forgatás;
         * - homályosságvizsgálat;
         * - kontraszt és szoftveres fényerő;
         * - Gemini-feldolgozás.
         */
        galleryLauncher =
                registerForActivityResult(
                        new ActivityResultContracts.GetContent(),
                        uri -> {
                            if (uri == null) {
                                return;
                            }

                            if (cameraStarted
                                    || (fullCameraContainer != null
                                    && fullCameraContainer.getVisibility()
                                    == View.VISIBLE)) {
                                closeFullScreenCamera();
                            }

                            loadPrescriptionFromGallery(
                                    uri
                            );
                        }
                );
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_scan, container, false);

        scanBtn = view.findViewById(R.id.scan_Btn);
        galleryBtn = view.findViewById(R.id.gallery_Btn);
        resultTextView = view.findViewById(R.id.result_TextView);
        receptImageView = view.findViewById(R.id.receptImageView);
        sliderContrast = view.findViewById(R.id.slider_contrast);
        btnFlash = view.findViewById(R.id.btn_flash);

        viewFinder = view.findViewById(R.id.viewFinder);
        tvCameraStatus = view.findViewById(R.id.tvCameraStatus);
        btnTakePhotoFull = view.findViewById(R.id.btnTakePhotoFull);
        btnCloseCamera = view.findViewById(R.id.btnCloseCamera);
        levelIndicator = view.findViewById(R.id.levelIndicator);
        fullCameraContainer = view.findViewById(R.id.fullCameraContainer);

        setupLevelIndicator();

        // 1. CSÚSZKA FIGYELÉSE - ÉLŐ FRISSÍTÉSSEL
        if (sliderContrast != null) {
            sliderContrast.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    contrastProgress = progress;
                    if (hasPhotoBeenTaken && baseBitmap != null) {
                        // Ha már van képünk, a csúszka mozdítására AZONNAL újraszámoljuk a filtert
                        applyLiveFilters();
                    }
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {
                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                }
            });
        }

        // 2. VAKU (FÉNYERŐ) FIGYELÉSE - ÉLŐ FRISSÍTÉSSEL
        if (btnFlash != null) {
            btnFlash.setOnClickListener(v -> {
                isFlashSimulated = !isFlashSimulated;
                if (isFlashSimulated) {
                    btnFlash.setBackgroundTintList(android.content.res.ColorStateList.valueOf(android.graphics.Color.YELLOW));
                } else {
                    btnFlash.setBackgroundTintList(android.content.res.ColorStateList.valueOf(android.graphics.Color.DKGRAY));
                }

                if (hasPhotoBeenTaken && baseBitmap != null) {
                    applyLiveFilters();
                }
            });
        }

        // 3. ENGEDÉLYKÉRŐ KEZELŐ
        requestPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                isGranted -> {
                    if (isGranted) {
                        openCamera();
                    } else {
                        Toast.makeText(getContext(), "Kamera engedély megtagadva!", Toast.LENGTH_LONG).show();
                    }
                }
        );

        // 4. TELJES KÉPERNYŐS KAMERA GOMBOK
        if (btnTakePhotoFull != null) {
            btnTakePhotoFull.setEnabled(false);
            btnTakePhotoFull.setOnClickListener(v -> takePhotoIfPhoneIsLevel());
        }

        if (btnCloseCamera != null) {
            btnCloseCamera.setOnClickListener(v -> closeFullScreenCamera());
        }

        // 5. KÉP KIVÁLASZTÁSA A GALÉRIÁBÓL
        if (galleryBtn != null) {
            galleryBtn.setOnClickListener(v -> {
                if (isTakingPhoto) {
                    Toast.makeText(
                            getContext(),
                            "A fénykép készítése még folyamatban van.",
                            Toast.LENGTH_SHORT
                    ).show();
                    return;
                }

                galleryLauncher.launch(
                        "image/*"
                );
            });
        }

        // 6. A FŐGOMB KETTŐS MŰKÖDÉSE
        scanBtn.setOnClickListener(v -> {
            if (!hasPhotoBeenTaken) {
                // HA MÉG NINCS KÉP (Vagy homályos lett) -> Elindítja a kamerát
                if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    requestPermissionLauncher.launch(Manifest.permission.CAMERA);
                } else {
                    openCamera();
                }
            } else {
                // HA MÁR VAN KÉP (És éles) -> Elküldi a beállított, módosított képet a Gemini-nek!
                if (processedBitmap != null) {
                    sendProcessedImageToGemini();
                }
            }
        });

        return view;
    }

    /**
     * A galériából kiválasztott receptképet háttérszálon tölti be, hogy
     * egy nagy felbontású kép se akassza meg a felületet.
     */
    private void loadPrescriptionFromGallery(
            @NonNull Uri uri
    ) {
        if (resultTextView != null) {
            resultTextView.setText(
                    "Receptkép betöltése a galériából..."
            );
        }

        if (scanBtn != null) {
            scanBtn.setEnabled(false);
        }

        if (galleryBtn != null) {
            galleryBtn.setEnabled(false);
        }

        cameraExecutor.execute(() -> {
            try {
                Bitmap selectedBitmap =
                        getSafeResizedBitmap(
                                uri
                        );

                if (selectedBitmap == null) {
                    throw new IOException(
                            "A kiválasztott képből nem hozható létre Bitmap."
                    );
                }

                handler.post(() -> {
                    if (!isAdded()
                            || getView() == null) {
                        if (!selectedBitmap.isRecycled()) {
                            selectedBitmap.recycle();
                        }
                        return;
                    }

                    if (scanBtn != null) {
                        scanBtn.setEnabled(true);
                    }

                    if (galleryBtn != null) {
                        galleryBtn.setEnabled(true);
                    }

                    handleGalleryBitmap(
                            selectedBitmap
                    );
                });

            } catch (Exception loadError) {
                Log.e(
                        "MedScanAI",
                        "A galériából választott receptkép betöltése sikertelen.",
                        loadError
                );

                handler.post(() -> {
                    if (!isAdded()
                            || getView() == null) {
                        return;
                    }

                    if (scanBtn != null) {
                        scanBtn.setEnabled(true);
                    }

                    if (galleryBtn != null) {
                        galleryBtn.setEnabled(true);
                    }

                    if (resultTextView != null) {
                        resultTextView.setText(
                                "A galériakép betöltése nem sikerült: "
                                        + loadError.getMessage()
                        );
                    }
                });
            }
        });
    }

    /**
     * A galériából származó képet nem menti el újra a
     * Download/MedScan/Images/Prescriptions mappába, mert az eredeti kép
     * már eleve megtalálható a felhasználó tárhelyén.
     */
    private void handleGalleryBitmap(
            @NonNull Bitmap bitmap
    ) {
        preparePrescriptionBitmap(
                bitmap,
                false
        );
    }

    private void openCamera() {
        openFullScreenCamera();
    }

    private void openFullScreenCamera() {
        if (fullCameraContainer == null) {
            if (resultTextView != null) {
                resultTextView.setText("Hiányzik az XML-ből a fullCameraContainer.");
            }
            return;
        }

        if (viewFinder == null) {
            if (resultTextView != null) {
                resultTextView.setText("Hiányzik az XML-ből a viewFinder.");
            }
            return;
        }

        if (accelerometer == null) {
            if (resultTextView != null) {
                resultTextView.setText("Ezen a készüléken nem érhető el a gyorsulásmérő szenzor.");
            }
            return;
        }

        fullCameraContainer.setVisibility(View.VISIBLE);

        if (tvCameraStatus != null) {
            tvCameraStatus.setText("Igazítsd vízszintbe a telefont.");
        }

        if (btnTakePhotoFull != null) {
            btnTakePhotoFull.setEnabled(false);
        }

        if (resultTextView != null) {
            resultTextView.setText("Kamera megnyitva. Zöld jelzésnél készíthető fotó.");
        }

        startCamera();
    }

    private void closeFullScreenCamera() {
        if (fullCameraContainer != null) {
            fullCameraContainer.setVisibility(View.GONE);
        }

        stopCamera();

        cameraStarted = false;
        imageCapture = null;
        isTakingPhoto = false;

        if (btnTakePhotoFull != null) {
            btnTakePhotoFull.setEnabled(false);
        }

        if (btnCloseCamera != null) {
            btnCloseCamera.setEnabled(true);
        }

        if (scanBtn != null) {
            scanBtn.setEnabled(true);

            if (!hasPhotoBeenTaken) {
                scanBtn.setText("KAMERA");
            }
        }

        if (galleryBtn != null) {
            galleryBtn.setEnabled(true);
        }

        if (resultTextView != null && !hasPhotoBeenTaken) {
            resultTextView.setText("Kamera bezárva.");
        }
    }

    private void setupLevelIndicator() {
        updateLevelIndicator(false);
    }

    private void updateLevelIndicator(boolean level) {
        if (levelIndicator == null) return;

        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(level ? android.graphics.Color.GREEN : android.graphics.Color.RED);
        drawable.setStroke(dpToPx(3), android.graphics.Color.WHITE);

        levelIndicator.setBackground(drawable);
        levelIndicator.setAlpha(0.95f);
    }

    private void startCamera() {
        ProcessCameraProvider.getInstance(requireContext()).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(requireContext()).get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                provider.unbindAll();

                provider.bindToLifecycle(
                        getViewLifecycleOwner(),
                        cameraSelector,
                        preview,
                        imageCapture
                );

                cameraStarted = true;

                if (tvCameraStatus != null) {
                    tvCameraStatus.setText("Igazítsd vízszintbe a telefont.");
                }

            } catch (Exception e) {
                if (resultTextView != null) {
                    resultTextView.setText("CameraX hiba: " + e.getMessage());
                }
                Log.e("MedScanAI", "CameraX hiba", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    private void takePhotoIfPhoneIsLevel() {
        if (isTakingPhoto) {
            return;
        }

        if (!isPhoneLevel) {
            if (tvCameraStatus != null) {
                tvCameraStatus.setText("Nem készíthető kép. Igazítsd vízszintbe a telefont.");
            }

            if (resultTextView != null) {
                resultTextView.setText("Nem készíthető kép! Tartsd vízszintesen a telefont, amíg a kör zöld nem lesz.");
            }
            return;
        }

        if (imageCapture == null) {
            if (resultTextView != null) {
                resultTextView.setText("A kamera még nem áll készen.");
            }
            return;
        }

        isTakingPhoto = true;

        if (btnTakePhotoFull != null) {
            btnTakePhotoFull.setEnabled(false);
        }

        if (btnCloseCamera != null) {
            btnCloseCamera.setEnabled(false);
        }

        if (scanBtn != null) {
            scanBtn.setEnabled(false);
        }

        if (galleryBtn != null) {
            galleryBtn.setEnabled(false);
        }

        if (tvCameraStatus != null) {
            tvCameraStatus.setText("Fénykép készítése...");
        }

        if (resultTextView != null) {
            resultTextView.setText("Fénykép készítése...");
        }

        imageCapture.takePicture(
                cameraExecutor,
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        Bitmap rotatedBitmap;

                        try {
                            Bitmap bitmap = imageProxyToBitmap(image);
                            int rotationDegrees = image.getImageInfo().getRotationDegrees();
                            rotatedBitmap = rotateBitmap(bitmap, rotationDegrees);
                        } catch (Exception e) {
                            image.close();

                            handler.post(() -> {
                                if (!isAdded() || getView() == null) return;

                                isTakingPhoto = false;

                                if (btnTakePhotoFull != null) {
                                    btnTakePhotoFull.setEnabled(isPhoneLevel);
                                }

                                if (btnCloseCamera != null) {
                                    btnCloseCamera.setEnabled(true);
                                }

                                if (scanBtn != null) {
                                    scanBtn.setEnabled(true);
                                }

                                if (galleryBtn != null) {
                                    galleryBtn.setEnabled(true);
                                }

                                if (resultTextView != null) {
                                    resultTextView.setText("Képfeldolgozási hiba: " + e.getMessage());
                                }
                            });

                            return;
                        }

                        image.close();

                        Bitmap finalRotatedBitmap = rotatedBitmap;

                        handler.post(() -> {
                            if (!isAdded() || getView() == null) return;

                            if (fullCameraContainer != null) {
                                fullCameraContainer.setVisibility(View.GONE);
                            }

                            if (btnCloseCamera != null) {
                                btnCloseCamera.setEnabled(true);
                            }

                            if (scanBtn != null) {
                                scanBtn.setEnabled(true);
                            }

                            if (galleryBtn != null) {
                                galleryBtn.setEnabled(true);
                            }

                            isTakingPhoto = false;
                            cameraStarted = false;
                            imageCapture = null;

                            stopCamera();

                            handleCapturedBitmap(finalRotatedBitmap);
                        });
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        handler.post(() -> {
                            if (!isAdded() || getView() == null) return;

                            isTakingPhoto = false;

                            if (btnTakePhotoFull != null) {
                                btnTakePhotoFull.setEnabled(isPhoneLevel);
                            }

                            if (btnCloseCamera != null) {
                                btnCloseCamera.setEnabled(true);
                            }

                            if (scanBtn != null) {
                                scanBtn.setEnabled(true);
                            }

                            if (galleryBtn != null) {
                                galleryBtn.setEnabled(true);
                            }

                            if (resultTextView != null) {
                                resultTextView.setText("Fotózási hiba: " + exception.getMessage());
                            }
                        });
                    }
                }
        );
    }

    /**
     * A kamerával készített receptfotó belépési pontja.
     */
    private void handleCapturedBitmap(
            Bitmap bitmap
    ) {
        preparePrescriptionBitmap(
                bitmap,
                true
        );
    }

    /**
     * A kamerából és a galériából származó receptképek közös
     * előkészítése.
     *
     * @param saveAsCapturedPhoto true esetén a frissen fotózott eredeti
     * képet külön is elmenti a receptmappába;
     * galériás képnél false.
     */
    private void preparePrescriptionBitmap(
            Bitmap bitmap,
            boolean saveAsCapturedPhoto
    ) {
        if (bitmap == null) {
            if (resultTextView != null) {
                resultTextView.setText(
                        "Nem sikerült receptképet betölteni."
                );
            }
            return;
        }

        if (bitmap.isRecycled()) {
            if (resultTextView != null) {
                resultTextView.setText(
                        "A kiválasztott receptkép már nem használható."
                );
            }
            return;
        }

        try {
            Bitmap previousBaseBitmap =
                    baseBitmap;

            Bitmap previousProcessedBitmap =
                    processedBitmap;

            /*
             * A nyers, még nem szűrt kép lesz az új szerkesztési alap.
             */
            baseBitmap =
                    bitmap;

            processedBitmap =
                    null;

            contrastProgress =
                    50;

            isFlashSimulated =
                    false;

            if (sliderContrast != null) {
                sliderContrast.setProgress(
                        50
                );
            }

            if (btnFlash != null) {
                btnFlash.setBackgroundTintList(
                        android.content.res.ColorStateList
                                .valueOf(
                                        android.graphics.Color.DKGRAY
                                )
                );
            }

            /*
             * Csak a közvetlenül kamerával készített fotót mentjük újra.
             * A galériából választott fájl már létezik a tárhelyen.
             */
            if (saveAsCapturedPhoto) {
                saveCapturedPrescriptionPhoto(
                        baseBitmap
                );
            }

            double blurScore =
                    calculateBlurScore(
                            baseBitmap
                    );

            Log.d(
                    "MedScanAI",
                    "Kép élességi pontszáma: "
                            + blurScore
                            + "; forrás="
                            + (
                            saveAsCapturedPhoto
                                    ? "camera"
                                    : "gallery"
                    )
            );

            /*
             * A kép rögtön megjelenik, így a felhasználó ugyanúgy
             * állíthatja a kontrasztot és a fényerőt mindkét forrásnál.
             */
            applyLiveFilters();

            if (blurScore < 100.0) {
                hasPhotoBeenTaken =
                        false;

                if (scanBtn != null) {
                    scanBtn.setText(
                            "Kép újra fotózása"
                    );

                    scanBtn.setBackgroundTintList(
                            android.content.res.ColorStateList
                                    .valueOf(
                                            android.graphics.Color.RED
                                    )
                    );
                }

                if (resultTextView != null) {
                    resultTextView.setText(
                            "⚠️ A kép túlságosan homályos a pontos "
                                    + "felismeréshez! Fotózd újra, vagy "
                                    + "válassz másik képet a galériából."
                    );
                }

            } else {
                hasPhotoBeenTaken =
                        true;

                if (scanBtn != null) {
                    scanBtn.setText(
                            "AI ELEMZÉS"
                    );

                    scanBtn.setBackgroundTintList(
                            android.content.res.ColorStateList
                                    .valueOf(
                                            android.graphics.Color
                                                    .parseColor(
                                                            "#2196F3"
                                                    )
                                    )
                    );
                }

                if (resultTextView != null) {
                    resultTextView.setText(
                            saveAsCapturedPhoto
                                    ? "✅ A kép éles és használható! "
                                      + "Finomhangolhatod a csúszkával, "
                                      + "majd küldd el az AI-nak."
                                    : "✅ A galériából kiválasztott kép "
                                      + "éles és használható! "
                                      + "Finomhangolhatod a csúszkával, "
                                      + "majd küldd el az AI-nak."
                    );
                }
            }

            /*
             * Az előző szerkesztési munkamenet képeinek felszabadítása.
             */
            if (previousProcessedBitmap != null
                    && previousProcessedBitmap != processedBitmap
                    && previousProcessedBitmap != bitmap
                    && !previousProcessedBitmap.isRecycled()) {
                previousProcessedBitmap.recycle();
            }

            if (previousBaseBitmap != null
                    && previousBaseBitmap != baseBitmap
                    && previousBaseBitmap != previousProcessedBitmap
                    && !previousBaseBitmap.isRecycled()) {
                previousBaseBitmap.recycle();
            }

        } catch (Exception error) {
            Log.e(
                    "MedScanAI",
                    "Hiba a receptkép előkészítésekor",
                    error
            );

            if (resultTextView != null) {
                resultTextView.setText(
                        "Hiba a receptkép betöltésekor: "
                                + error.getMessage()
                );
            }
        }
    }

    /**
     * Háttérszálon elmenti a frissen készített, eredeti receptfotót.
     *
     * Android 10 vagy újabb:
     *   Download/MedScan/Images0/
     *
     * Android 7–9:
     *   Android/data/<csomagnév>/files/Download/MedScan/Images/Prescriptions/
     *
     * A mentés nem függ a Gemini eredményétől.
     */
    private void saveCapturedPrescriptionPhoto(
            @NonNull Bitmap originalPrescriptionBitmap
    ) {
        if (originalPrescriptionBitmap.isRecycled()) {
            Log.e(
                    "PrescriptionPhotoSave",
                    "A receptfotó már fel lett szabadítva."
            );
            return;
        }

        /*
         * A meglévő cameraExecutor használható, mert a fénykép elkészítése
         * után a CameraX-kép már le lett zárva. Így a nagy JPEG-mentés nem
         * blokkolja a főszálat és a felületet.
         */
        cameraExecutor.execute(() -> {
            try {
                Uri savedUri;

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    savedUri = savePrescriptionWithMediaStore(
                            originalPrescriptionBitmap
                    );
                } else {
                    savedUri = savePrescriptionToAppPictures(
                            originalPrescriptionBitmap
                    );
                }

                Log.d(
                        "PrescriptionPhotoSave",
                        "Eredeti receptfotó elmentve: " + savedUri
                );

            } catch (Exception saveError) {
                Log.e(
                        "PrescriptionPhotoSave",
                        "Az eredeti receptfotó mentése sikertelen.",
                        saveError
                );

                handler.post(() -> {
                    if (!isAdded() || getContext() == null) {
                        return;
                    }

                    Toast.makeText(
                            getContext(),
                            "A receptfotó külön mappába mentése nem sikerült.",
                            Toast.LENGTH_LONG
                    ).show();
                });
            }
        });
    }

    /**
     * Android 10 vagy újabb rendszeren a kép a felhasználó által is látható
     * Download/MedScan/Images/Prescriptions mappába kerül.
     */
    @RequiresApi(api = Build.VERSION_CODES.Q)
    private Uri savePrescriptionWithMediaStore(
            @NonNull Bitmap bitmap
    ) throws IOException {
        String filename =
                createPrescriptionFilename();

        ContentValues values = new ContentValues();
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
                        + "/MedScan/Images/Prescriptions"
        );
        values.put(
                MediaStore.MediaColumns.IS_PENDING,
                1
        );

        Uri imageUri = requireContext()
                .getContentResolver()
                .insert(
                        MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                        values
                );

        if (imageUri == null) {
            throw new IOException(
                    "A MediaStore nem hozta létre a receptfotót."
            );
        }

        boolean completed = false;

        try (OutputStream outputStream =
                     requireContext()
                             .getContentResolver()
                             .openOutputStream(imageUri)) {
            if (outputStream == null) {
                throw new IOException(
                        "A receptfotó kimeneti adatfolyama nem nyitható meg."
                );
            }

            boolean compressed = bitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    95,
                    outputStream
            );

            if (!compressed) {
                throw new IOException(
                        "A receptfotó JPEG-tömörítése sikertelen."
                );
            }

            outputStream.flush();
            completed = true;

        } finally {
            if (!completed) {
                requireContext()
                        .getContentResolver()
                        .delete(
                                imageUri,
                                null,
                                null
                        );
            }
        }

        ContentValues finishedValues = new ContentValues();
        finishedValues.put(
                MediaStore.MediaColumns.IS_PENDING,
                0
        );

        requireContext()
                .getContentResolver()
                .update(
                        imageUri,
                        finishedValues,
                        null,
                        null
                );

        return imageUri;
    }

    /**
     * Android 7–9 rendszeren az alkalmazás saját külső Download mappájába
     * mentünk. Ehhez nem szükséges WRITE_EXTERNAL_STORAGE engedély.
     */
    private Uri savePrescriptionToAppPictures(
            @NonNull Bitmap bitmap
    ) throws IOException {
        File downloadsRoot =
                requireContext().getExternalFilesDir(
                        Environment.DIRECTORY_DOWNLOADS
                );

        if (downloadsRoot == null) {
            throw new IOException(
                    "A külső Download mappa nem érhető el."
            );
        }

        File prescriptionDirectory =
                new File(
                        downloadsRoot,
                        "MedScan/Images/Prescriptions"
                );

        if (!prescriptionDirectory.exists()
                && !prescriptionDirectory.mkdirs()) {
            throw new IOException(
                    "A MedScan/Images/Prescriptions mappa nem hozható létre."
            );
        }

        File outputFile =
                new File(
                        prescriptionDirectory,
                        createPrescriptionFilename()
                );

        boolean completed = false;

        try (FileOutputStream outputStream =
                     new FileOutputStream(outputFile)) {
            boolean compressed = bitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    95,
                    outputStream
            );

            if (!compressed) {
                throw new IOException(
                        "A receptfotó JPEG-tömörítése sikertelen."
                );
            }

            outputStream.flush();
            completed = true;

        } finally {
            if (!completed && outputFile.exists()) {
                //noinspection ResultOfMethodCallIgnored
                outputFile.delete();
            }
        }

        return Uri.fromFile(outputFile);
    }

    private String createPrescriptionFilename() {
        String timestamp =
                new SimpleDateFormat(
                        "yyyyMMdd_HHmmss_SSS",
                        Locale.getDefault()
                ).format(new Date());

        return "prescription_" + timestamp + ".jpg";
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

    private Bitmap imageProxyToBitmap(ImageProxy image) throws Exception {
        if (image.getPlanes() == null || image.getPlanes().length == 0) {
            throw new Exception("Nincs kép adat.");
        }

        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        if (bitmap == null) {
            throw new Exception("Nem sikerült Bitmap képet létrehozni.");
        }

        return bitmap;
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
    public void onSensorChanged(SensorEvent event) {
        if (event == null || event.sensor == null || event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) {
            return;
        }

        float rawX = event.values[0];
        float rawY = event.values[1];
        float rawZ = event.values[2];

        /*
         * Szenzor simítás.
         * Ez azért kell, hogy idősebbeknél / kézremegésnél ne villogjon gyorsan
         * a zöld-piros jelzés.
         */
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

        /*
         * Dőlés számítása fokban.
         * A telefon akkor megfelelő, ha X és Y irányban is maximum ±15 fokban tér el.
         */
        double tiltX = Math.toDegrees(Math.atan2(filteredX, filteredZ));
        double tiltY = Math.toDegrees(Math.atan2(filteredY, filteredZ));

        isPhoneLevel =
                Math.abs(tiltX) <= LEVEL_TOLERANCE_DEGREES &&
                        Math.abs(tiltY) <= LEVEL_TOLERANCE_DEGREES &&
                        filteredZ > MIN_FACE_UP_Z;

        updateLevelIndicator(isPhoneLevel);

        if (cameraStarted && !isTakingPhoto) {
            if (isPhoneLevel) {
                if (tvCameraStatus != null) {
                    tvCameraStatus.setText("Telefon megfelelő helyzetben. Készíthető fotó.");
                }

                if (btnTakePhotoFull != null) {
                    btnTakePhotoFull.setEnabled(true);
                }
            } else {
                if (tvCameraStatus != null) {
                    tvCameraStatus.setText("Igazítsd vízszintbe a telefont. Megengedett eltérés: ±15°.");
                }

                if (btnTakePhotoFull != null) {
                    btnTakePhotoFull.setEnabled(false);
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
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
        }

        if (fullCameraContainer != null) {
            fullCameraContainer.setVisibility(View.GONE);
        }

        isTakingPhoto = false;
    }

    // Élő képfrissítés a felületen a ColorMatrix segítségével
    private void applyLiveFilters() {
        if (baseBitmap == null) return;

        try {
            // Kontraszt kiszámítása (50 a középérték)
            float contrast = (float) contrastProgress / 50.0f;
            float brightness = isFlashSimulated ? 40f : 0f;

            // BIZTONSÁGI JAVÍTÁS: Ha a kép túl nagy, lekicsinyítjük, nehogy fekete legyen a kijelző
            Bitmap workingBitmap = baseBitmap;
            if (baseBitmap.getWidth() > 2048 || baseBitmap.getHeight() > 2048) {
                int newWidth = 1280;
                int newHeight = (int) ((float) baseBitmap.getHeight() * ((float) 1280 / (float) baseBitmap.getWidth()));
                workingBitmap = Bitmap.createScaledBitmap(baseBitmap, newWidth, newHeight, true);
                Log.d("MedScanAI", "Kép átméretezve a biztonságos kirajzoláshoz: " + newWidth + "x" + newHeight);
            }

            // Létrehozzuk a szűrt bitmapet a megfelelő méretben
            Bitmap previousProcessedBitmap =
                    processedBitmap;

            Bitmap.Config bitmapConfig =
                    workingBitmap.getConfig() == null
                            ? Bitmap.Config.ARGB_8888
                            : workingBitmap.getConfig();

            processedBitmap =
                    Bitmap.createBitmap(
                            workingBitmap.getWidth(),
                            workingBitmap.getHeight(),
                            bitmapConfig
                    );

            Canvas canvas = new Canvas(processedBitmap);
            Paint paint = new Paint();

            // ColorMatrix a kontraszthoz és a fényerőhöz
            ColorMatrix cm = new ColorMatrix(new float[] {
                    contrast, 0, 0, 0, brightness + (128f * (1f - contrast)),
                    0, contrast, 0, 0, brightness + (128f * (1f - contrast)),
                    0, 0, contrast, 0, brightness + (128f * (1f - contrast)),
                    0, 0, 0, 1f, 0
            });

            paint.setColorFilter(new ColorMatrixColorFilter(cm));
            canvas.drawBitmap(workingBitmap, 0, 0, paint);

            // Kényszerítjük a hardveres gyorsítást, hogy azonnal jelenjen meg a kép
            receptImageView.setLayerType(View.LAYER_TYPE_SOFTWARE, null);
            receptImageView.setBackgroundColor(android.graphics.Color.TRANSPARENT);
            receptImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            receptImageView.setImageBitmap(processedBitmap);

            if (previousProcessedBitmap != null
                    && previousProcessedBitmap != processedBitmap
                    && previousProcessedBitmap != baseBitmap
                    && !previousProcessedBitmap.isRecycled()) {
                previousProcessedBitmap.recycle();
            }

            if (workingBitmap != baseBitmap
                    && !workingBitmap.isRecycled()) {
                workingBitmap.recycle();
            }

            Log.d("MedScanAI", "A szűrt kép sikeresen kirajzolva az ImageView-ba.");

        } catch (OutOfMemoryError oom) {
            Log.e("MedScanAI", "Kifutott a memória a képfeldolgozás során!", oom);
            // Ha mégis elfogyna a memória, tartalékként a nyers képet rakjuk be szűrés nélkül, így nem lesz fekete
            receptImageView.setImageBitmap(baseBitmap);
        } catch (Exception e) {
            Log.e("MedScanAI", "Hiba a szűrő alkalmazásakor", e);
        }
    }

    /*
     * Az előzményekbe az eredeti, még kontraszt- és fényerőmódosítás
     * előtti receptfotó kerül.
     *
     * A Gemini továbbra is a felhasználó által beállított processedBitmap
     * képet elemzi, tehát a receptfelismerési működés nem változik.
     */
    private void sendProcessedImageToGemini() {
        if (baseBitmap == null) {
            if (resultTextView != null) {
                resultTextView.setText(
                        "Hiba: az eredeti receptfotó nem található."
                );
            }
            return;
        }

        if (processedBitmap == null) {
            if (resultTextView != null) {
                resultTextView.setText(
                        "Hiba: a feldolgozott receptkép nem található."
                );
            }
            return;
        }

        File originalFile = null;

        try {
            File historyDirectory = new File(
                    requireContext().getFilesDir(),
                    "history_images"
            );

            if (!historyDirectory.exists()
                    && !historyDirectory.mkdirs()) {
                throw new IOException(
                        "Az előzményképek mappája nem hozható létre."
                );
            }

            originalFile = new File(
                    historyDirectory,
                    "prescription_original_"
                            + System.currentTimeMillis()
                            + ".jpg"
            );

            try (FileOutputStream outputStream =
                         new FileOutputStream(originalFile)) {
                boolean saved = baseBitmap.compress(
                        Bitmap.CompressFormat.JPEG,
                        95,
                        outputStream
                );

                if (!saved) {
                    throw new IOException(
                            "Az eredeti receptfotó mentése sikertelen."
                    );
                }

                outputStream.flush();
            }

            Uri originalImageUri =
                    Uri.fromFile(originalFile);

            hasPhotoBeenTaken = false;
            scanBtn.setText("KAMERA");

            /*
             * Az URI az előzményekhez tartozik.
             * A bitmap továbbra is a módosított AI-bemenet.
             */
            processImageWithGemini(
                    originalImageUri,
                    processedBitmap
            );

        } catch (Exception error) {
            Log.e(
                    "MedScanAI",
                    "Hiba az eredeti receptfotó mentésekor",
                    error
            );

            if (originalFile != null
                    && originalFile.exists()
                    && originalFile.length() == 0L) {
                //noinspection ResultOfMethodCallIgnored
                originalFile.delete();
            }

            if (resultTextView != null) {
                resultTextView.setText(
                        "Hiba az eredeti receptfotó mentésekor: "
                                + error.getMessage()
                );
            }
        }
    }

    private void processImageWithGemini(Uri currentImageUri, Bitmap bitmap) {
        // 1. UI előkészítése a főszálon
        if (resultTextView != null) {
            resultTextView.setText("AI feldolgozás és strukturálás folyamatban...");
        }

        try {
            GenerativeModel gm = new GenerativeModel("gemini-2.5-flash", "AQ.Ab8RN6Le2p2tHzZBS5_7GXvU9HQz_jdGHk5_XGJy7h9IR5s_2g");
            GenerativeModelFutures model = GenerativeModelFutures.from(gm);

            String prompt =
                    "Te egy magyar egészségügyi receptadat-feldolgozó rendszer vagy. "
                            + "Olvasd le a recept képéről a beteg nevét és a gyógyszer adatait. "
                            + "Nagyon fontos, hogy a BETEG nevét és a FELÍRÓ ORVOS nevét "
                            + "külön kezeld. "
                            + "A patientName kizárólag annak a betegnek a teljes neve lehet, "
                            + "akinek a receptet kiállították. A beteg nevét a Beteg neve, "
                            + "Páciens neve, Biztosított neve vagy a betegadatokat tartalmazó "
                            + "rész alapján keresd. "
                            + "A doctorName kizárólag a felíró orvos neve lehet, amely gyakran "
                            + "orvosi pecsét, aláírás, Felíró orvos, Orvos neve vagy "
                            + "orvosazonosító közelében található. "
                            + "SOHA ne másold a doctorName értékét a patientName mezőbe. "
                            + "Ha nem tudod teljes bizonyossággal elkülöníteni a beteg nevét "
                            + "az orvos nevétől, a patientName értéke legyen üres string. "
                            + "A name mező kizárólag a gyógyszer neve lehet, ember neve nem. "
                            + "Javítsd ki az egyértelmű optikai elgépeléseket a magyar nyelv "
                            + "logikája alapján, de ne találj ki hiányzó adatot. "
                            + "KIZÁRÓLAG egy érvényes JSON objektumot adj vissza, "
                            + "felesleges szöveg és markdown formázás nélkül. "
                            + "A JSON pontosan ezeket a kulcsokat tartalmazza: "
                            + "\"patientName\", \"doctorName\", \"name\", \"weight\", "
                            + "\"amount\", \"weightUnit\", \"portion\". "
                            + "Ha egy szöveges adat nem olvasható biztosan, az értéke legyen "
                            + "üres string, az amount pedig 0.";

            Content content = new Content.Builder()
                    .addText(prompt)
                    .addImage(bitmap)
                    .build();

            ListenableFuture<GenerateContentResponse> responseFuture = model.generateContent(content);

            Futures.addCallback(responseFuture, new FutureCallback<GenerateContentResponse>() {
                @Override
                public void onSuccess(GenerateContentResponse result) {
                    String rawResult = result.getText();
                    if (rawResult == null) return;
                    rawResult = rawResult.replace("```json", "").replace("```", "").trim();

                    try {
                        JSONObject jsonObject =
                                new JSONObject(rawResult);

                        String gainedName =
                                jsonObject.optString(
                                        "name",
                                        ""
                                ).trim();

                        String rawPatientName =
                                jsonObject.optString(
                                        "patientName",
                                        ""
                                ).trim();

                        String doctorName =
                                jsonObject.optString(
                                        "doctorName",
                                        ""
                                ).trim();

                        String patientName =
                                sanitizePatientName(
                                        rawPatientName,
                                        doctorName
                                );

                        int amount =
                                jsonObject.optInt(
                                        "amount",
                                        0
                                );

                        String portion =
                                jsonObject.optString(
                                        "portion",
                                        ""
                                ).trim();

                        String weight =
                                jsonObject.optString(
                                        "weight",
                                        ""
                                ).trim();

                        String weightUnit =
                                jsonObject.optString(
                                        "weightUnit",
                                        ""
                                ).trim();

                        String displayedMedicineName =
                                gainedName.isEmpty()
                                        ? "Ismeretlen"
                                        : gainedName;

                        String displayedPatientName =
                                patientName.isEmpty()
                                        ? "Nem olvasható biztosan"
                                        : patientName;

                        String displayedPortion =
                                portion.isEmpty()
                                        ? "Ismeretlen"
                                        : portion;

                        String displayedWeight =
                                weight.isEmpty()
                                        ? "Ismeretlen"
                                        : weight;

                        String displayedWeightUnit =
                                weightUnit.isEmpty()
                                        ? ""
                                        : weightUnit;

                        // 2. UI frissítése biztonságosan a főszálon
                        if (getActivity() != null) {
                            getActivity().runOnUiThread(() -> {
                                resultTextView.setText(
                                        "Beteg neve: "
                                                + displayedPatientName
                                                + "\nGyógyszer neve: "
                                                + displayedMedicineName
                                                + "\nMennyiség: "
                                                + amount
                                                + " db"
                                                + "\nBevétel: "
                                                + displayedPortion
                                                + "\nHatóanyag tartalom: "
                                                + displayedWeight
                                                + (displayedWeightUnit.isEmpty()
                                                ? ""
                                                : " " + displayedWeightUnit)
                                );
                            });
                        }

                        // Adatbázis mentés
                        java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyy. MM. dd. HH:mm:ss", java.util.Locale.getDefault());
                        String formazottDatum = sdf.format(new java.util.Date());

                        String currentUserId = "ismeretlen_felhasznalo";
                        if (com.google.firebase.auth.FirebaseAuth.getInstance().getCurrentUser() != null) {
                            currentUserId = com.google.firebase.auth.FirebaseAuth.getInstance().getCurrentUser().getUid();
                        }

                        Recept newRecept = new Recept(
                                gainedName,
                                amount,
                                portion,
                                formazottDatum,
                                weight.isEmpty()
                                        ? null
                                        : weight,
                                weightUnit,
                                currentImageUri.toString(),
                                currentUserId
                        );

                        /*
                         * A Firestore-ban külön patientName mezőként tároljuk.
                         * A doctorName szándékosan nem kerül mentésre.
                         */
                        newRecept.setPatientName(
                                patientName
                        );

                        newRecept.setHistoryType(
                                Recept.TYPE_PRESCRIPTION
                        );

                        FirebaseFirestore db = FirebaseFirestore.getInstance();
                        db.collection("receptek")
                                .add(newRecept)
                                .addOnSuccessListener(doc -> {
                                    if (isAdded() && getContext() != null) {
                                        Toast.makeText(getContext(), "Sikeresen mentve az adatbázisba!", Toast.LENGTH_SHORT).show();
                                    }
                                })
                                .addOnFailureListener(e -> Log.e("MedScanAI", "Firestore hiba", e));

                    } catch (Exception ex) {
                        Log.e("MedScanAI", "JSON hiba", ex);
                        if (getActivity() != null) {
                            getActivity().runOnUiThread(() -> resultTextView.setText("Hiba az adatok feldolgozásakor."));
                        }
                    }
                }

                @Override
                public void onFailure(@NonNull Throwable t) {
                    Log.e("MedScanAI", "Gemini hiba", t);
                    if (getActivity() != null) {
                        getActivity().runOnUiThread(() -> resultTextView.setText("Hiba az AI hívásakor."));

                    }
                }
            }, ContextCompat.getMainExecutor(requireContext()));

        } catch (Exception e) {
            Log.e("MedScanAI", "Hiba a kép előkészítésekor", e);
        }
    }

    /**
     * Megakadályozza, hogy a Gemini által orvosként is felismert név
     * betegnévként kerüljön a Firestore-ba.
     */
    private String sanitizePatientName(
            String patientName,
            String doctorName
    ) {
        String patient =
                patientName == null
                        ? ""
                        : patientName.trim();

        String doctor =
                doctorName == null
                        ? ""
                        : doctorName.trim();

        if (patient.isEmpty()) {
            return "";
        }

        String normalizedPatient =
                normalizePersonNameForComparison(
                        patient
                );

        String normalizedDoctor =
                normalizePersonNameForComparison(
                        doctor
                );

        if (normalizedPatient.isEmpty()) {
            return "";
        }

        /*
         * Ha ugyanazt a nevet adta vissza betegként és orvosként,
         * inkább nem tárolunk bizonytalan betegnevet.
         */
        if (!normalizedDoctor.isEmpty()
                && normalizedPatient.equals(
                normalizedDoctor
        )) {
            Log.w(
                    "MedScanAI",
                    "A beteg és az orvos neve azonosnak tűnt, "
                            + "ezért a patientName nem kerül mentésre."
            );
            return "";
        }

        /*
         * Tipikus orvosi megjelölések és pecsétadatok kizárása.
         */
        if (normalizedPatient.startsWith("dr ")
                || normalizedPatient.equals("dr")
                || normalizedPatient.contains(" doktor")
                || normalizedPatient.contains(" orvos")
                || normalizedPatient.contains("pecsetszam")
                || normalizedPatient.contains("orvosazonosito")
                || normalizedPatient.contains("feliro orvos")) {
            Log.w(
                    "MedScanAI",
                    "Orvosra utaló érték került a patientName mezőbe, "
                            + "ezért nem mentjük el: "
                            + patient
            );
            return "";
        }

        /*
         * A személynévben szereplő számjegy általában TAJ-számra,
         * pecsétszámra vagy egyéb azonosítóra utal.
         */
        for (int i = 0; i < patient.length(); i++) {
            if (Character.isDigit(
                    patient.charAt(i)
            )) {
                Log.w(
                        "MedScanAI",
                        "Számjegyet tartalmazó patientName elutasítva: "
                                + patient
                );
                return "";
            }
        }

        if ("ismeretlen".equals(normalizedPatient)
                || "n a".equals(normalizedPatient)
                || "nincs adat".equals(normalizedPatient)
                || "nem olvashato".equals(normalizedPatient)) {
            return "";
        }

        return patient;
    }

    /**
     * Ékezetektől, írásjelektől és többszörös szóközöktől független
     * összehasonlító alakot készít a személynevekből.
     */
    private String normalizePersonNameForComparison(
            String value
    ) {
        if (value == null
                || value.trim().isEmpty()) {
            return "";
        }

        return Normalizer.normalize(
                        value,
                        Normalizer.Form.NFD
                )
                .replaceAll(
                        "\\p{M}+",
                        ""
                )
                .toLowerCase(
                        Locale.ROOT
                )
                .replaceAll(
                        "[^a-z\\s]",
                        " "
                )
                .replaceAll(
                        "\\s+",
                        " "
                )
                .trim();
    }

    private Bitmap getSafeResizedBitmap(Uri uri) throws IOException {
        if (getContext() == null) return null;

        // 1. Alap bitmap beolvasása tömörítve
        java.io.InputStream input = getContext().getContentResolver().openInputStream(uri);
        android.graphics.BitmapFactory.Options options = new android.graphics.BitmapFactory.Options();
        options.inSampleSize = 2; // Méret felezése a memóriavédelem miatt
        Bitmap bitmap = android.graphics.BitmapFactory.decodeStream(input, null, options);
        if (input != null) input.close();

        if (bitmap == null) return null;

        // 2. EXIF adatok lekérése a forgatási szög meghatározásához
        int eloforgatas = 0;
        try {
            java.io.InputStream exifInput = getContext().getContentResolver().openInputStream(uri);
            if (exifInput != null) {
                androidx.exifinterface.media.ExifInterface exifInterface =
                        new androidx.exifinterface.media.ExifInterface(exifInput);

                int orientation = exifInterface.getAttributeInt(
                        androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL);

                switch (orientation) {
                    case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90:
                        eloforgatas = 90;
                        break;
                    case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180:
                        eloforgatas = 180;
                        break;
                    case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270:
                        eloforgatas = 270;
                        break;
                }
                exifInput.close();
            }
        } catch (Exception e) {
            Log.e("MedScanAI", "Nem sikerült leolvasni az EXIF forgatási adatokat", e);
        }

        // 3. Ha a telefon elforgatta a képet, szoftveresen visszaforgatjuk a helyes irányba
        if (eloforgatas != 0) {
            android.graphics.Matrix matrix = new android.graphics.Matrix();
            matrix.postRotate(eloforgatas);
            Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

            // Felszabadítjuk a régi elforgatott bitmapet a memóriából
            if (rotatedBitmap != bitmap) {
                bitmap.recycle();
            }
            Log.d("MedScanAI", "Kép sikeresen visszaforgatva: " + eloforgatas + " fokkal.");
            return rotatedBitmap;
        }

        return bitmap;
    }

    // Szoftveres élességvizsgáló (Laplace variancia alapján)
    private double calculateBlurScore(Bitmap bitmap) {
        if (bitmap == null) return 0;

        // A sebesség érdekében lekicsinyítjük a képet az elemzéshez
        int width = 300;
        int height = (int) (300 * ((float) bitmap.getHeight() / bitmap.getWidth()));
        Bitmap smallBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true);

        int[] pixels = new int[width * height];
        smallBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // Szürkeárnyalatosra alakítás a gyorsabb számolásért
        int[] gray = new int[width * height];
        for (int i = 0; i < pixels.length; i++) {
            int r = (pixels[i] >> 16) & 0xFF;
            int g = (pixels[i] >> 8) & 0xFF;
            int b = pixels[i] & 0xFF;
            gray[i] = (int) (0.299 * r + 0.587 * g + 0.114 * b);
        }

        // 3x3 Laplace élkereső mátrix alkalmazása
        long sum = 0;
        long sumSq = 0;
        int count = 0;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int val = -4 * gray[y * width + x]
                        + gray[(y - 1) * width + x]
                        + gray[(y + 1) * width + x]
                        + gray[y * width + (x - 1)]
                        + gray[y * width + (x + 1)];
                sum += val;
                sumSq += (long) val * val;
                count++;
            }
        }

        // Variancia (élesség) kiszámítása
        double mean = (double) sum / count;
        return ((double) sumSq / count) - (mean * mean);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();

        scanBtn = null;
        galleryBtn = null;
        resultTextView = null;
        receptImageView = null;
        sliderContrast = null;
        btnFlash = null;

        viewFinder = null;
        tvCameraStatus = null;
        btnTakePhotoFull = null;
        btnCloseCamera = null;
        levelIndicator = null;
        fullCameraContainer = null;

        imageCapture = null;

        if (processedBitmap != null
                && processedBitmap != baseBitmap
                && !processedBitmap.isRecycled()) {
            processedBitmap.recycle();
        }

        if (baseBitmap != null
                && !baseBitmap.isRecycled()) {
            baseBitmap.recycle();
        }

        processedBitmap = null;
        baseBitmap = null;
        hasPhotoBeenTaken = false;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    private int dpToPx(int dp) {
        return Math.round(dp * getResources().getDisplayMetrics().density);
    }
}