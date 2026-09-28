package com.example.kmd5a9_szakmaigyakorlat;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;

/**
 * A négy ArUco-marker belső sarkai közötti mező perspektívakorrekciója.
 *
 * Markerelrendezés a nyomtatott lapon, óramutató járásával megegyezően:
 * 0 = bal felső
 * 1 = jobb felső
 * 2 = jobb alsó
 * 3 = bal alsó
 */
public final class ArUcoProcessor {

    private static final String TAG = "ArUcoProcessor";

    /*
     * A markerek belső sarkai közötti tényleges mező mérete.
     * A jelenlegi lap alapján: 150 mm széles, 90 mm magas.
     */
    private static final double INNER_FIELD_WIDTH_MM = 150.0;
    private static final double INNER_FIELD_HEIGHT_MM = 90.0;

    /*
     * 6 px/mm esetén a kimenet 900x540 px.
     * Ez elég részletes, de gyorsabb a korábbi 1500x900-as képnél.
     */
    private static final double PIXELS_PER_MILLIMETER = 6.0;

    private static final int ID_TOP_LEFT = 0;
    private static final int ID_TOP_RIGHT = 1;
    private static final int ID_BOTTOM_RIGHT = 2;
    private static final int ID_BOTTOM_LEFT = 3;

    private ArUcoProcessor() {
    }

    public static Mat processSheet(Mat input) throws Exception {
        if (input == null || input.empty()) {
            throw new Exception(
                    "Az ArUco-feldolgozás bemeneti képe üres."
            );
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

        List<Mat> detectedCorners =
                new ArrayList<>();

        Mat ids = new Mat();

        try {
            detector.detectMarkers(
                    input,
                    detectedCorners,
                    ids
            );

            Point[][] cornersById =
                    new Point[4][];

            for (int row = 0; row < ids.rows(); row++) {
                double[] idData = ids.get(row, 0);

                if (idData == null
                        || idData.length == 0) {
                    continue;
                }

                int id =
                        (int) Math.round(idData[0]);

                if (id < 0 || id > 3) {
                    continue;
                }

                if (row >= detectedCorners.size()) {
                    continue;
                }

                cornersById[id] =
                        readMarkerCorners(
                                detectedCorners.get(row)
                        );
            }

            for (int requiredId = 0;
                 requiredId <= 3;
                 requiredId++) {

                if (cornersById[requiredId] == null) {
                    throw new Exception(
                            "Hiányzik a szükséges ArUco-marker. ID="
                                    + requiredId
                    );
                }
            }

            /*
             * OpenCV saroksorrend:
             * 0 = bal felső
             * 1 = jobb felső
             * 2 = jobb alsó
             * 3 = bal alsó
             *
             * Csak a középső mező felé néző belső sarkokat használjuk:
             * ID 0 (bal felső)  -> jobb alsó sarok, index 2
             * ID 1 (jobb felső) -> bal alsó sarok, index 3
             * ID 2 (jobb alsó)  -> bal felső sarok, index 0
             * ID 3 (bal alsó)   -> jobb felső sarok, index 1
             */
            Point innerTopLeft =
                    cornersById[ID_TOP_LEFT][2];

            Point innerTopRight =
                    cornersById[ID_TOP_RIGHT][3];

            Point innerBottomRight =
                    cornersById[ID_BOTTOM_RIGHT][0];

            Point innerBottomLeft =
                    cornersById[ID_BOTTOM_LEFT][1];

            int outputWidth =
                    Math.max(
                            1,
                            (int) Math.round(
                                    INNER_FIELD_WIDTH_MM
                                            * PIXELS_PER_MILLIMETER
                            )
                    );

            int outputHeight =
                    Math.max(
                            1,
                            (int) Math.round(
                                    INNER_FIELD_HEIGHT_MM
                                            * PIXELS_PER_MILLIMETER
                            )
                    );

            MatOfPoint2f source =
                    new MatOfPoint2f(
                            innerTopLeft,
                            innerTopRight,
                            innerBottomRight,
                            innerBottomLeft
                    );

            MatOfPoint2f destination =
                    new MatOfPoint2f(
                            new Point(0, 0),
                            new Point(outputWidth - 1, 0),
                            new Point(
                                    outputWidth - 1,
                                    outputHeight - 1
                            ),
                            new Point(0, outputHeight - 1)
                    );

            Mat transform = null;

            try {
                transform =
                        Imgproc.getPerspectiveTransform(
                                source,
                                destination
                        );

                Mat output = new Mat();

                Imgproc.warpPerspective(
                        input,
                        output,
                        transform,
                        new Size(
                                outputWidth,
                                outputHeight
                        ),
                        Imgproc.INTER_CUBIC,
                        Core.BORDER_CONSTANT,
                        new Scalar(
                                145,
                                145,
                                145,
                                255
                        )
                );

                Log.d(
                        TAG,
                        "ArUco-korrekció kész. "
                                + "ID sorrend: 0=bal felső, 1=jobb felső, "
                                + "2=jobb alsó, 3=bal alsó. "
                                + "Csak a belső sarkok közötti mező maradt. "
                                + "Kimenet="
                                + outputWidth
                                + "x"
                                + outputHeight
                );

                return output;

            } finally {
                source.release();
                destination.release();

                if (transform != null) {
                    transform.release();
                }
            }

        } finally {
            ids.release();

            for (Mat corners : detectedCorners) {
                if (corners != null) {
                    corners.release();
                }
            }
        }
    }

    private static Point[] readMarkerCorners(
            Mat markerCorners
    ) throws Exception {
        if (markerCorners == null
                || markerCorners.empty()
                || markerCorners.cols() < 4) {
            throw new Exception(
                    "Hibás ArUco sarokadat."
            );
        }

        Point[] result = new Point[4];

        for (int index = 0; index < 4; index++) {
            double[] coordinate =
                    markerCorners.get(0, index);

            if (coordinate == null
                    || coordinate.length < 2) {
                throw new Exception(
                        "Nem olvasható az ArUco-marker egyik sarka."
                );
            }

            result[index] =
                    new Point(
                            coordinate[0],
                            coordinate[1]
                    );
        }

        return result;
    }

    public static double getInnerFieldWidthMillimeters() {
        return INNER_FIELD_WIDTH_MM;
    }

    public static double getInnerFieldHeightMillimeters() {
        return INNER_FIELD_HEIGHT_MM;
    }

    public static double getPixelsPerMillimeter() {
        return PIXELS_PER_MILLIMETER;
    }
}
