package com.example.kmd5a9_szakmaigyakorlat;

import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.CameraInfo;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;

/**
 * A kamera és az ArUco-lap közötti merőleges távolság becslése.
 *
 * A becsléshez:
 * - a négy ArUco-marker belső sarkait;
 * - a belső mező ismert, 150 x 90 mm-es fizikai méretét;
 * - a telefon kamerájának Camera2 kalibrációs adatait
 * használjuk.
 *
 * A visszaadott érték a kamera optikai középpontjának a lap síkjától
 * mért becsült magassága centiméterben.
 */
public final class ArUcoHeightEstimator {

    private static final String TAG = "ArUcoHeightEstimator";

    private static final int ID_TOP_LEFT = 0;
    private static final int ID_TOP_RIGHT = 1;
    private static final int ID_BOTTOM_RIGHT = 2;
    private static final int ID_BOTTOM_LEFT = 3;

    private static final double MIN_VALID_HEIGHT_CM = 5.0;
    private static final double MAX_VALID_HEIGHT_CM = 150.0;
    private static final double MAX_WIDTH_HEIGHT_DISAGREEMENT_RATIO = 0.30;

    private ArUcoHeightEstimator() {
    }

    /**
     * A CameraX által ténylegesen használt kamerából kiolvassa a szükséges
     * Camera2 kalibrációs adatokat.
     */
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    public static CameraCalibration createCalibration(
            @NonNull CameraInfo cameraInfo
    ) throws Exception {
        Camera2CameraInfo camera2Info = Camera2CameraInfo.from(cameraInfo);

        Rect activeArray = camera2Info.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
        );
        Size pixelArray = camera2Info.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE
        );
        SizeF physicalSize = camera2Info.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE
        );
        float[] focalLengths = camera2Info.getCameraCharacteristic(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
        );
        float[] intrinsicCalibration = camera2Info.getCameraCharacteristic(
                CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
        );

        if (activeArray == null || activeArray.width() <= 0 || activeArray.height() <= 0) {
            throw new Exception("A kamera aktív szenzorterülete nem olvasható.");
        }

        double fxSensor;
        double fySensor;
        double cxSensor;
        double cySensor;
        String source;

        /*
         * Elsődlegesen a gyártó által megadott intrinsics értékeket használjuk:
         * [fx, fy, cx, cy, skew].
         */
        if (intrinsicCalibration != null
                && intrinsicCalibration.length >= 4
                && intrinsicCalibration[0] > 0f
                && intrinsicCalibration[1] > 0f) {

            fxSensor = intrinsicCalibration[0];
            fySensor = intrinsicCalibration[1];
            cxSensor = intrinsicCalibration[2] - activeArray.left;
            cySensor = intrinsicCalibration[3] - activeArray.top;
            source = "LENS_INTRINSIC_CALIBRATION";

        } else {
            /*
             * Fallback: fókusztávolság (mm), fizikai szenzorméret (mm) és
             * pixelszám alapján számítjuk a fókusztávolságot pixelben.
             */
            if (pixelArray == null
                    || physicalSize == null
                    || physicalSize.getWidth() <= 0f
                    || physicalSize.getHeight() <= 0f
                    || focalLengths == null
                    || focalLengths.length == 0
                    || focalLengths[0] <= 0f) {
                throw new Exception(
                        "A kamera nem ad elegendő kalibrációs adatot a magasságméréshez."
                );
            }

            double focalLengthMm = focalLengths[0];

            fxSensor = focalLengthMm
                    / physicalSize.getWidth()
                    * pixelArray.getWidth();

            fySensor = focalLengthMm
                    / physicalSize.getHeight()
                    * pixelArray.getHeight();

            cxSensor = pixelArray.getWidth() / 2.0 - activeArray.left;
            cySensor = pixelArray.getHeight() / 2.0 - activeArray.top;
            source = "FOCAL_LENGTH_AND_SENSOR_SIZE";
        }

        if (!isFinitePositive(fxSensor) || !isFinitePositive(fySensor)) {
            throw new Exception("Érvénytelen kamerakalibrációs fókuszadat.");
        }

        CameraCalibration calibration = new CameraCalibration(
                activeArray.width(),
                activeArray.height(),
                fxSensor,
                fySensor,
                cxSensor,
                cySensor,
                source
        );

        Log.d(
                TAG,
                "Kamerakalibráció betöltve. Forrás=" + source
                        + "; aktív szenzor="
                        + activeArray.width() + "x" + activeArray.height()
                        + "; fx=" + fxSensor
                        + "; fy=" + fySensor
        );

        return calibration;
    }

    /**
     * Megkeresi a 0,1,2,3 ArUco-kódot, majd az ismert belső mezőből
     * megbecsüli a kamera lap feletti magasságát.
     */
    public static HeightMeasurement estimateHeight(
            Mat image,
            CameraCalibration calibration
    ) {
        if (image == null || image.empty()) {
            return HeightMeasurement.invalid(
                    false,
                    "Az elemzett kamerakép üres."
            );
        }

        Dictionary dictionary = Objdetect.getPredefinedDictionary(
                Objdetect.DICT_4X4_50
        );
        ArucoDetector detector = new ArucoDetector(
                dictionary,
                new DetectorParameters()
        );

        List<Mat> detectedCorners = new ArrayList<>();
        Mat ids = new Mat();

        /*
         * Külön tároljuk, hogy a négy szükséges marker felismerése
         * sikerült-e. Így egy későbbi magasságmérési vagy kalibrációs
         * hiba nem állítja vissza tévesen a markerállapotot false-ra.
         */
        boolean allRequiredMarkersFound = false;

        try {
            detector.detectMarkers(image, detectedCorners, ids);

            Point[][] cornersById = new Point[4][];

            for (int row = 0; row < ids.rows(); row++) {
                double[] idValue = ids.get(row, 0);
                if (idValue == null || idValue.length == 0) {
                    continue;
                }

                int id = (int) Math.round(idValue[0]);
                if (id < 0 || id > 3 || row >= detectedCorners.size()) {
                    continue;
                }

                cornersById[id] = readMarkerCorners(detectedCorners.get(row));
            }

            for (int id = 0; id <= 3; id++) {
                if (cornersById[id] == null) {
                    return HeightMeasurement.invalid(
                            false,
                            "Nem látható mind a négy szükséges ArUco-kód."
                    );
                }
            }

            /*
             * Ettől a ponttól biztosan megvan a 0, 1, 2 és 3 marker.
             */
            allRequiredMarkersFound = true;

            /*
             * A markerfelismerés nem függhet a kamerakalibrációtól.
             * Kalibráció nélkül a magasság nem számítható ki, de a
             * keret ettől még bezöldülhet, mert a négy marker megvan.
             */
            if (calibration == null) {
                return HeightMeasurement.invalid(
                        true,
                        "Nincs kamerakalibráció."
                );
            }

            Point innerTopLeft = cornersById[ID_TOP_LEFT][2];
            Point innerTopRight = cornersById[ID_TOP_RIGHT][3];
            Point innerBottomRight = cornersById[ID_BOTTOM_RIGHT][0];
            Point innerBottomLeft = cornersById[ID_BOTTOM_LEFT][1];

            double topWidthPx = distance(innerTopLeft, innerTopRight);
            double bottomWidthPx = distance(innerBottomLeft, innerBottomRight);
            double leftHeightPx = distance(innerTopLeft, innerBottomLeft);
            double rightHeightPx = distance(innerTopRight, innerBottomRight);

            double averageWidthPx = (topWidthPx + bottomWidthPx) / 2.0;
            double averageHeightPx = (leftHeightPx + rightHeightPx) / 2.0;

            if (averageWidthPx < 20.0 || averageHeightPx < 20.0) {
                return HeightMeasurement.invalid(
                        true,
                        "Az ArUco-mező túl kicsi a megbízható méréshez."
                );
            }

            ImageIntrinsics imageIntrinsics = calibration.forImageSize(
                    image.cols(),
                    image.rows()
            );

            double widthHeightMm = ArUcoProcessor.getInnerFieldWidthMillimeters();
            double fieldHeightMm = ArUcoProcessor.getInnerFieldHeightMillimeters();

            double heightFromWidthMm = imageIntrinsics.fx
                    * widthHeightMm
                    / averageWidthPx;

            double heightFromHeightMm = imageIntrinsics.fy
                    * fieldHeightMm
                    / averageHeightPx;

            if (!isFinitePositive(heightFromWidthMm)
                    || !isFinitePositive(heightFromHeightMm)) {
                return HeightMeasurement.invalid(
                        true,
                        "A számított magasság nem érvényes."
                );
            }

            double disagreementRatio = Math.abs(
                    heightFromWidthMm - heightFromHeightMm
            ) / Math.max(heightFromWidthMm, heightFromHeightMm);

            if (disagreementRatio > MAX_WIDTH_HEIGHT_DISAGREEMENT_RATIO) {
                return HeightMeasurement.invalid(
                        true,
                        "A lap túl ferdén látszik a megbízható magasságméréshez."
                );
            }

            /*
             * A két független becslés átlagát használjuk. A telefon vízszintes
             * helyzetét külön a gyorsulásmérő ellenőrzi.
             */
            double heightCm = (
                    heightFromWidthMm + heightFromHeightMm
            ) / 20.0;

            if (heightCm < MIN_VALID_HEIGHT_CM
                    || heightCm > MAX_VALID_HEIGHT_CM
                    || Double.isNaN(heightCm)
                    || Double.isInfinite(heightCm)) {
                return HeightMeasurement.invalid(
                        true,
                        "A mért magasság kívül esik az érvényes tartományon."
                );
            }

            return HeightMeasurement.valid(
                    heightCm,
                    heightFromWidthMm / 10.0,
                    heightFromHeightMm / 10.0,
                    calibration.source
            );

        } catch (Exception e) {
            Log.w(TAG, "ArUco-magasságmérési hiba", e);
            return HeightMeasurement.invalid(
                    allRequiredMarkersFound,
                    "Magasságmérési hiba: " + e.getMessage()
            );
        } finally {
            ids.release();
            for (Mat corners : detectedCorners) {
                if (corners != null) {
                    corners.release();
                }
            }
        }
    }

    private static Point[] readMarkerCorners(Mat markerCorners) throws Exception {
        if (markerCorners == null
                || markerCorners.empty()
                || markerCorners.cols() < 4) {
            throw new Exception("Hibás ArUco sarokadat.");
        }

        Point[] result = new Point[4];
        for (int index = 0; index < 4; index++) {
            double[] coordinate = markerCorners.get(0, index);
            if (coordinate == null || coordinate.length < 2) {
                throw new Exception("Nem olvasható az ArUco-marker egyik sarka.");
            }

            result[index] = new Point(coordinate[0], coordinate[1]);
        }
        return result;
    }

    private static double distance(Point a, Point b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private static boolean isFinitePositive(double value) {
        return value > 0.0
                && !Double.isNaN(value)
                && !Double.isInfinite(value);
    }

    public static final class CameraCalibration {
        private final int sensorWidth;
        private final int sensorHeight;
        private final double fxSensor;
        private final double fySensor;
        private final double cxSensor;
        private final double cySensor;
        private final String source;

        private CameraCalibration(
                int sensorWidth,
                int sensorHeight,
                double fxSensor,
                double fySensor,
                double cxSensor,
                double cySensor,
                String source
        ) {
            this.sensorWidth = sensorWidth;
            this.sensorHeight = sensorHeight;
            this.fxSensor = fxSensor;
            this.fySensor = fySensor;
            this.cxSensor = cxSensor;
            this.cySensor = cySensor;
            this.source = source;
        }

        /**
         * A szenzorkoordinátákat a CameraX ImageAnalysis aktuális képére
         * skálázza. Eltérő képaránynál középre igazított cropot feltételez.
         */
        private ImageIntrinsics forImageSize(int imageWidth, int imageHeight) {
            double scale = Math.max(
                    imageWidth / (double) sensorWidth,
                    imageHeight / (double) sensorHeight
            );

            double scaledSensorWidth = sensorWidth * scale;
            double scaledSensorHeight = sensorHeight * scale;

            double cropX = (scaledSensorWidth - imageWidth) / 2.0;
            double cropY = (scaledSensorHeight - imageHeight) / 2.0;

            return new ImageIntrinsics(
                    fxSensor * scale,
                    fySensor * scale,
                    cxSensor * scale - cropX,
                    cySensor * scale - cropY
            );
        }
    }

    private static final class ImageIntrinsics {
        private final double fx;
        private final double fy;
        @SuppressWarnings("unused")
        private final double cx;
        @SuppressWarnings("unused")
        private final double cy;

        private ImageIntrinsics(double fx, double fy, double cx, double cy) {
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
        }
    }

    public static final class HeightMeasurement {
        private final boolean allMarkersFound;
        private final boolean valid;
        private final double heightCm;
        private final double widthBasedHeightCm;
        private final double heightBasedHeightCm;
        private final String calibrationSource;
        private final String message;

        private HeightMeasurement(
                boolean allMarkersFound,
                boolean valid,
                double heightCm,
                double widthBasedHeightCm,
                double heightBasedHeightCm,
                String calibrationSource,
                String message
        ) {
            this.allMarkersFound = allMarkersFound;
            this.valid = valid;
            this.heightCm = heightCm;
            this.widthBasedHeightCm = widthBasedHeightCm;
            this.heightBasedHeightCm = heightBasedHeightCm;
            this.calibrationSource = calibrationSource;
            this.message = message;
        }

        public static HeightMeasurement valid(
                double heightCm,
                double widthBasedHeightCm,
                double heightBasedHeightCm,
                String calibrationSource
        ) {
            return new HeightMeasurement(
                    true,
                    true,
                    heightCm,
                    widthBasedHeightCm,
                    heightBasedHeightCm,
                    calibrationSource,
                    "OK"
            );
        }

        public static HeightMeasurement invalid(
                boolean allMarkersFound,
                String message
        ) {
            return new HeightMeasurement(
                    allMarkersFound,
                    false,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    "",
                    message == null ? "Ismeretlen mérési hiba." : message
            );
        }

        public boolean areAllMarkersFound() {
            return allMarkersFound;
        }

        public boolean isValid() {
            return valid;
        }

        public double getHeightCm() {
            return heightCm;
        }

        public double getWidthBasedHeightCm() {
            return widthBasedHeightCm;
        }

        public double getHeightBasedHeightCm() {
            return heightBasedHeightCm;
        }

        public String getCalibrationSource() {
            return calibrationSource;
        }

        public String getMessage() {
            return message;
        }
    }
}