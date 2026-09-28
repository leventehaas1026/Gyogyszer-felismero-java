package com.example.kmd5a9_szakmaigyakorlat;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Teljesen helyi és determinisztikus gyógyszerszegmentáló.
 *
 * Ugyanazon bemeneti Bitmap esetén:
 * - nincs hálózati kérés;
 * - nincs generatív modell;
 * - nincs véletlen mintavételezés;
 * - ugyanazok az OpenCV-műveletek ugyanabban a sorrendben futnak;
 * - ezért ugyanaz a bináris maszk készül.
 *
 * Az algoritmus a perspektívakorrigált ArUco-mező világos papírhátteréből
 * indul ki. A háttér színét a kép külső sávjából becsüli, majd kombinálja:
 *
 * 1. a Lab-színtérbeli háttérkülönbséget;
 * 2. a fényességkülönbséget;
 * 3. a Canny-éleket;
 * 4. a morfológiai zárást és nyitást;
 * 5. a geometriai kontúrszűrést;
 * 6. a gyógyszer alakját stabilizáló konvex burkot.
 */
public final class DeterministicPillSegmenter {

    private static final String TAG =
            "DeterministicPillSegmenter";

    public static final String MODEL_NAME =
            "opencv-deterministic-lab-v9-local-max-instance-watershed";

    /*
     * Ezek fix paraméterek. A determinisztikusság egyik feltétele, hogy
     * futtatásonként ne változzanak.
     */
    private static final double BORDER_RATIO = 0.075;
    private static final double MIN_COMPONENT_AREA_RATIO = 0.00035;
    private static final double MAX_COMPONENT_AREA_RATIO = 0.22;
    private static final double MIN_SOLIDITY = 0.24;
    private static final double MAX_ASPECT_RATIO = 6.0;

    private static final int MIN_COMPONENT_SIDE_PX = 12;
    private static final int BORDER_REJECTION_PX = 5;

    private static final int SCORE_BLUR_SIZE = 5;
    private static final int EDGE_DILATION_SIZE = 3;
    private static final int MORPH_CLOSE_SIZE = 5;
    private static final int MORPH_OPEN_SIZE = 3;

    /*
     * A kerek/ovális tabletták determinisztikus alak-kiegészítésének
     * biztonsági határai.
     *
     * A kiegészítés csak akkor fut le, ha a megtalált kontúr már eleve
     * tablettaszerű, az illesztett ellipszis pedig nem növeli meg
     * indokolatlanul az objektumot.
     */
    /*
     * Közel kör alakú tablettáknál nem a hiányos kontúrra illesztett
     * ellipszis az elsődleges, hanem a legkisebb befoglaló kör.
     */
    /*
     * A körbesoroláshoz nem az axis-aligned boundingRect arányát
     * használjuk, mert egy ferdén álló hosszúkás gyógyszer téglalapja
     * közel négyzetes lehet. A minAreaRect elforgatásfüggetlen arányát
     * vizsgáljuk.
     */
    private static final double
            MAX_CIRCLE_COMPLETION_ORIENTED_ASPECT_RATIO = 1.60;

    private static final double MIN_CIRCLE_COMPLETION_SOLIDITY = 0.38;
    private static final double MIN_CIRCLE_TO_HULL_AREA_RATIO = 0.95;
    private static final double MAX_CIRCLE_TO_HULL_AREA_RATIO = 2.15;
    private static final double MAX_CIRCLE_DIAMETER_TO_RECT_RATIO = 1.72;
    private static final double CIRCLE_COMPLETION_RADIUS_PADDING = 1.035;

    /*
     * A tervezett kör kerületének jelentős részén valódi Canny-élnek
     * kell lennie. Egy hosszúkás tabletta köré rajzolt kör ezt nem
     * teljesíti, egy hiányos körtabletta viszont igen.
     */
    private static final int CIRCLE_EDGE_SAMPLE_COUNT = 180;
    private static final double MIN_CIRCLE_EDGE_SUPPORT_RATIO = 0.38;
    private static final int MIN_CIRCLE_EDGE_SUPPORTED_QUADRANTS = 3;

    private static final double MAX_COMPLETION_ASPECT_RATIO = 2.35;
    private static final double MIN_COMPLETION_SOLIDITY = 0.55;
    private static final double MIN_ELLIPSE_TO_HULL_AREA_RATIO = 0.96;
    private static final double MAX_ELLIPSE_TO_HULL_AREA_RATIO = 1.58;
    private static final double MAX_ELLIPSE_AXIS_GROWTH = 1.55;
    private static final double MAX_ELLIPSE_CENTER_OFFSET_RATIO = 0.30;

    /*
     * A végleges alakot nem a hibás, V alakban lezárt convex hullból
     * becsüljük. A durva maszk csak a gyógyszer helyét jelöli ki,
     * majd a közeli valódi Canny-élekre elforgatott ellipszist
     * illesztünk.
     */
    private static final double EDGE_INFLUENCE_RADIUS_RATIO = 0.16;
    private static final int MIN_EDGE_POINTS_FOR_ELLIPSE = 24;
    private static final double MIN_ELLIPSE_COARSE_COVERAGE = 0.66;
    private static final double MIN_ELLIPSE_AREA_RATIO = 0.72;
    private static final double MAX_ELLIPSE_AREA_RATIO = 2.35;
    private static final double MAX_ELLIPSE_AXIS_RATIO = 6.0;
    private static final double MAX_ELLIPSE_CENTER_SHIFT_RATIO = 0.48;
    private static final double ELLIPSE_PADDING_RATIO = 1.055;
    private static final double MIN_BOUNDARY_NORMALIZED_RADIUS = 0.72;
    private static final double MAX_BOUNDARY_NORMALIZED_RADIUS = 1.32;

    /*
     * GrabCut-alapú végső maszk.
     *
     * A durva maszk és az élillesztett ellipszis csak inicializálás.
     * A végleges alak a forráskép tényleges előtér-komponenséből készül,
     * majd szoros elforgatott ellipszissé stabilizálódik.
     */
    private static final int GRABCUT_ITERATIONS = 7;
    private static final double GRABCUT_ROI_SCALE = 1.65;
    private static final double GRABCUT_PROBABLE_FOREGROUND_SCALE = 1.28;
    private static final double GRABCUT_DEFINITE_FOREGROUND_SCALE = 0.48;
    private static final double GRABCUT_FINAL_ELLIPSE_SCALE = 0.990;
    private static final double GRABCUT_MIN_AXIS_RATIO_TO_INITIAL = 0.58;
    private static final double GRABCUT_MAX_AXIS_RATIO_TO_INITIAL = 1.34;
    private static final double GRABCUT_MAX_CENTER_SHIFT_RATIO = 0.34;
    private static final double GRABCUT_MIN_CORE_COVERAGE = 0.82;

    /*
     * Az árnyék a gyógyszer belső felületénél sötétebb, de a GrabCut
     * gyakran ugyanahhoz az összefüggő komponenshez sorolja.
     *
     * A belső biztos előtér L-csatorna átlaga alapján adaptív alsó
     * fényességhatárt számítunk. A küszöb nem fix képérték, ezért
     * világosabb és sötétebb gyógyszereknél is használható.
     */
    private static final double SHADOW_L_DROP_RATIO = 0.09;
    private static final double SHADOW_MIN_L_DROP = 12.0;
    private static final double SHADOW_MAX_L_DROP = 24.0;
    private static final double SHADOW_CLOSE_KERNEL_RATIO = 0.115;
    private static final int SHADOW_MIN_CLOSE_KERNEL = 3;
    private static final int SHADOW_MAX_CLOSE_KERNEL = 11;

    /*
     * Több gyógyszer szétválasztása.
     *
     * A morfológia vagy az árnyék miatt összekapcsolódó foltokat
     * distance transform + watershed bontja külön példányokra.
     */
    /*
     * Lokálismaximum-alapú watershed markerek.
     *
     * A korábbi küszöbölt seedMask több közeli gyógyszernél összeolvadt,
     * ezért egy gyógyszercsoportból csak 1-2 marker készült.
     *
     * Most a distance transform valódi lokális maximumait keressük.
     * Egy maximum egy gyógyszer belső magját jelöli.
     */
    private static final double WATERSHED_LOCAL_MAX_VALUE_RATIO = 0.18;
    private static final double WATERSHED_LOCAL_MAX_WINDOW_RATIO = 0.65;
    private static final int WATERSHED_LOCAL_MAX_MIN_RADIUS_PX = 3;
    private static final int WATERSHED_LOCAL_MAX_MAX_RADIUS_PX = 18;
    private static final int WATERSHED_MAX_INSTANCE_COUNT = 32;

    /*
     * Egy hosszúkás kapszula distance transformja több hasonló
     * maximumot is tartalmazhat. Ha két maximum között végig magas
     * distance-ridge fut, ugyanahhoz a gyógyszerhez tartoznak.
     */
    private static final double WATERSHED_RIDGE_MERGE_RATIO = 0.62;
    private static final double WATERSHED_NEAR_PEAK_RATIO = 0.70;

    /*
     * A kis watershed-régiókat se dobjuk el túl agresszíven.
     */
    private static final double WATERSHED_MIN_INSTANCE_AREA_RATIO = 0.12;
    private static final int WATERSHED_MIN_INSTANCE_AREA_PX = 18;
    private static final int WATERSHED_RESULT_CLOSE_SIZE = 3;

    /*
     * A végleges ellipszis/GrabCut maszk nem léphet át a saját
     * watershed-régiójából a szomszéd gyógyszerre.
     */
    private static final double INSTANCE_OWNER_MARGIN_RATIO = 0.055;
    private static final int INSTANCE_OWNER_MIN_MARGIN_PX = 2;
    private static final int INSTANCE_OWNER_MAX_MARGIN_PX = 7;

    private static final double MIN_FOREGROUND_THRESHOLD = 12.0;
    private static final double MAX_FOREGROUND_THRESHOLD = 52.0;

    /**
     * Több, egymástól elkülönülő gyógyszert is visszaad.
     *
     * A sorrend stabil:
     * - először fentről lefelé;
     * - azonos sorban balról jobbra.
     */
    public List<SegmentationResult> segmentMedicines(
            @NonNull Bitmap sourceBitmap
    ) throws Exception {
        validateBitmap(sourceBitmap);

        long startedAt =
                SystemClock.elapsedRealtime();

        /*
         * Bár ez a verzió nem használ véletlen OpenCV-műveletet,
         * rögzítjük az RNG állapotát, hogy későbbi bővítésnél se
         * jelenjen meg futtatásonkénti eltérés.
         */
        Core.setRNGSeed(20260729);

        Mat rgba = new Mat();
        Mat rgb = new Mat();
        Mat lab = new Mat();
        Mat gray = new Mat();
        Mat borderMask = null;
        Mat validPaperMask = new Mat();
        Mat sampleMask = new Mat();
        Mat lChannel = new Mat();

        Mat backgroundLab = null;
        Mat labDifference = new Mat();
        Mat grayDifference = new Mat();
        Mat foregroundScore = new Mat();
        Mat blurredScore = new Mat();

        Mat otsuMask = new Mat();
        Mat thresholdMask = new Mat();
        Mat edgeMask = new Mat();
        Mat combinedMask = new Mat();
        Mat cleanedMask = new Mat();

        Mat closeKernel = null;
        Mat openKernel = null;
        Mat edgeKernel = null;

        List<MatOfPoint> contours = new ArrayList<>();
        List<SegmentCandidate> candidates =
                new ArrayList<>();

        try {
            Utils.bitmapToMat(
                    sourceBitmap,
                    rgba
            );

            Imgproc.cvtColor(
                    rgba,
                    rgb,
                    Imgproc.COLOR_RGBA2RGB
            );

            /*
             * Egy kis mediánszűrés eltávolítja az egyedi szenzorzajt,
             * miközben a gyógyszerek éleit jobban megtartja, mint egy
             * nagy Gauss-szűrés.
             */
            Imgproc.medianBlur(
                    rgb,
                    rgb,
                    SCORE_BLUR_SIZE
            );

            Imgproc.cvtColor(
                    rgb,
                    lab,
                    Imgproc.COLOR_RGB2Lab
            );

            Imgproc.cvtColor(
                    rgb,
                    gray,
                    Imgproc.COLOR_RGB2GRAY
            );

            List<Mat> labChannels =
                    new ArrayList<>();

            Core.split(
                    lab,
                    labChannels
            );

            if (labChannels.size() != 3) {
                throw new Exception(
                        "A Lab-kép nem tartalmaz három csatornát."
                );
            }

            labChannels.get(0).copyTo(
                    lChannel
            );

            borderMask =
                    createBorderMask(
                            sourceBitmap.getWidth(),
                            sourceBitmap.getHeight()
                    );

            /*
             * A papír háttérmintájából kizárjuk a nagyon sötét
             * feliratokat és nyomtatott elemeket.
             */
            Imgproc.threshold(
                    lChannel,
                    validPaperMask,
                    135.0,
                    255.0,
                    Imgproc.THRESH_BINARY
            );

            Core.bitwise_and(
                    borderMask,
                    validPaperMask,
                    sampleMask
            );

            int sampleCount =
                    Core.countNonZero(
                            sampleMask
                    );

            if (sampleCount
                    < Math.max(
                    20,
                    sourceBitmap.getWidth()
                            * sourceBitmap.getHeight()
                            / 500
            )) {
                /*
                 * Szokatlanul sötét kép esetén a teljes külső sávból
                 * becsüljük a hátteret.
                 */
                borderMask.copyTo(
                        sampleMask
                );
            }

            Scalar paperLab =
                    Core.mean(
                            lab,
                            sampleMask
                    );

            Scalar paperGrayScalar =
                    Core.mean(
                            gray,
                            sampleMask
                    );

            backgroundLab =
                    new Mat(
                            lab.size(),
                            lab.type(),
                            new Scalar(
                                    paperLab.val[0],
                                    paperLab.val[1],
                                    paperLab.val[2]
                            )
                    );

            Core.absdiff(
                    lab,
                    backgroundLab,
                    labDifference
            );

            List<Mat> differenceChannels =
                    new ArrayList<>();

            Core.split(
                    labDifference,
                    differenceChannels
            );

            if (differenceChannels.size() != 3) {
                throw new Exception(
                        "A Lab-különbségkép hibás."
                );
            }

            /*
             * A három Lab-csatorna közül a legnagyobb eltérést használjuk.
             * Ez a színes gyógyszereknél és a fehér tabletta árnyékos
             * széleinél is stabil jelet ad.
             */
            Core.max(
                    differenceChannels.get(0),
                    differenceChannels.get(1),
                    foregroundScore
            );

            Core.max(
                    foregroundScore,
                    differenceChannels.get(2),
                    foregroundScore
            );

            Mat paperGray =
                    new Mat(
                            gray.size(),
                            gray.type(),
                            new Scalar(
                                    paperGrayScalar.val[0]
                            )
                    );

            try {
                Core.absdiff(
                        gray,
                        paperGray,
                        grayDifference
                );
            } finally {
                paperGray.release();
            }

            Core.max(
                    foregroundScore,
                    grayDifference,
                    foregroundScore
            );

            Imgproc.GaussianBlur(
                    foregroundScore,
                    blurredScore,
                    new Size(
                            SCORE_BLUR_SIZE,
                            SCORE_BLUR_SIZE
                    ),
                    0.0
            );

            double otsuThreshold =
                    Imgproc.threshold(
                            blurredScore,
                            otsuMask,
                            0.0,
                            255.0,
                            Imgproc.THRESH_BINARY
                                    | Imgproc.THRESH_OTSU
                    );

            double foregroundThreshold =
                    clamp(
                            otsuThreshold,
                            MIN_FOREGROUND_THRESHOLD,
                            MAX_FOREGROUND_THRESHOLD
                    );

            Imgproc.threshold(
                    blurredScore,
                    thresholdMask,
                    foregroundThreshold,
                    255.0,
                    Imgproc.THRESH_BINARY
            );

            /*
             * Az élmaszk a papírhoz hasonlóan világos gyógyszerek
             * külső körvonalát egészíti ki.
             */
            Imgproc.Canny(
                    gray,
                    edgeMask,
                    35.0,
                    105.0
            );

            edgeKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    EDGE_DILATION_SIZE,
                                    EDGE_DILATION_SIZE
                            )
                    );

            Imgproc.dilate(
                    edgeMask,
                    edgeMask,
                    edgeKernel
            );

            Core.bitwise_or(
                    thresholdMask,
                    edgeMask,
                    combinedMask
            );

            closeKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    MORPH_CLOSE_SIZE,
                                    MORPH_CLOSE_SIZE
                            )
                    );

            openKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    MORPH_OPEN_SIZE,
                                    MORPH_OPEN_SIZE
                            )
                    );

            Imgproc.morphologyEx(
                    combinedMask,
                    cleanedMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel,
                    new Point(-1, -1),
                    1
            );

            Imgproc.morphologyEx(
                    cleanedMask,
                    cleanedMask,
                    Imgproc.MORPH_OPEN,
                    openKernel,
                    new Point(-1, -1),
                    1
            );

            double imageArea =
                    (double) sourceBitmap.getWidth()
                            * sourceBitmap.getHeight();

            double minimumArea =
                    Math.max(
                            70.0,
                            imageArea
                                    * MIN_COMPONENT_AREA_RATIO
                    );

            double maximumArea =
                    imageArea
                            * MAX_COMPONENT_AREA_RATIO;

            /*
             * Nem a teljes cleanedMask külső kontúrjait tekintjük
             * automatikusan egy-egy gyógyszernek. Előbb külön példányokra
             * bontjuk az összekapcsolódó foltokat.
             */
            List<Mat> instanceMasks =
                    separateMedicineInstances(
                            rgb,
                            cleanedMask,
                            minimumArea
                    );

            try {
                for (Mat instanceMask : instanceMasks) {
                    if (instanceMask == null
                            || instanceMask.empty()) {
                        continue;
                    }

                    List<MatOfPoint> instanceContours =
                            new ArrayList<>();

                    Mat instanceCopy =
                            instanceMask.clone();

                    Mat instanceHierarchy =
                            new Mat();

                    try {
                        Imgproc.findContours(
                                instanceCopy,
                                instanceContours,
                                instanceHierarchy,
                                Imgproc.RETR_EXTERNAL,
                                Imgproc.CHAIN_APPROX_SIMPLE
                        );
                    } finally {
                        instanceCopy.release();
                        instanceHierarchy.release();
                    }

                    MatOfPoint instanceContour =
                            largestContour(
                                    instanceContours
                            );

                    if (instanceContour != null) {
                        SegmentCandidate candidate =
                                createCandidate(
                                        instanceContour,
                                        sourceBitmap.getWidth(),
                                        sourceBitmap.getHeight(),
                                        minimumArea,
                                        maximumArea,
                                        edgeMask,
                                        rgb,
                                        instanceMask
                                );

                        if (candidate != null) {
                            candidates.add(
                                    candidate
                            );
                        }
                    }

                    releaseContours(
                            instanceContours
                    );
                }
            } finally {
                for (Mat instanceMask : instanceMasks) {
                    if (instanceMask != null) {
                        instanceMask.release();
                    }
                }

                instanceMasks.clear();
            }

            if (candidates.isEmpty()) {
                throw new Exception(
                        "A determinisztikus OpenCV-s szegmentálás "
                                + "nem talált megfelelő gyógyszerkontúrt."
                );
            }

            /*
             * Stabil sorrend, hogy ugyanaz a gyógyszer mindig ugyanazt
             * az indexet kapja.
             */
            Collections.sort(
                    candidates,
                    Comparator
                            .comparingDouble(
                                    (SegmentCandidate value) ->
                                            value.centerY
                            )
                            .thenComparingDouble(
                                    value ->
                                            value.centerX
                            )
            );

            long runtimeMs =
                    SystemClock.elapsedRealtime()
                            - startedAt;

            List<SegmentationResult> results =
                    new ArrayList<>();

            int index = 1;

            for (SegmentCandidate candidate
                    : candidates) {
                JSONObject diagnostics =
                        new JSONObject();

                diagnostics.put(
                        "algorithm",
                        MODEL_NAME
                );
                diagnostics.put(
                        "deterministic",
                        true
                );
                diagnostics.put(
                        "segment_index",
                        index
                );
                diagnostics.put(
                        "input_width",
                        sourceBitmap.getWidth()
                );
                diagnostics.put(
                        "input_height",
                        sourceBitmap.getHeight()
                );
                diagnostics.put(
                        "background_lab",
                        new JSONArray()
                                .put(
                                        round(
                                                paperLab.val[0]
                                        )
                                )
                                .put(
                                        round(
                                                paperLab.val[1]
                                        )
                                )
                                .put(
                                        round(
                                                paperLab.val[2]
                                        )
                                )
                );
                diagnostics.put(
                        "otsu_threshold",
                        round(
                                otsuThreshold
                        )
                );
                diagnostics.put(
                        "used_threshold",
                        round(
                                foregroundThreshold
                        )
                );
                diagnostics.put(
                        "area_px",
                        round(
                                candidate.area
                        )
                );
                diagnostics.put(
                        "solidity",
                        round(
                                candidate.solidity
                        )
                );
                diagnostics.put(
                        "oriented_aspect_ratio",
                        round(
                                candidate.orientedAspectRatio
                        )
                );
                diagnostics.put(
                        "geometric_completion",
                        candidate.completionMode
                );
                diagnostics.put(
                        "bbox",
                        new JSONArray()
                                .put(
                                        candidate.box.left
                                )
                                .put(
                                        candidate.box.top
                                )
                                .put(
                                        candidate.box.right
                                )
                                .put(
                                        candidate.box.bottom
                                )
                );
                diagnostics.put(
                        "mask_sha256",
                        candidate.maskSha256
                );
                diagnostics.put(
                        "runtime_ms",
                        runtimeMs
                );

                results.add(
                        new SegmentationResult(
                                MODEL_NAME,
                                "medicine",
                                candidate.polygon,
                                candidate.box,
                                diagnostics.toString(2),
                                diagnostics.toString(2),
                                runtimeMs,
                                candidate.maskSha256
                        )
                );

                index++;
            }

            Log.d(
                    TAG,
                    String.format(
                            Locale.ROOT,
                            "Deterministic segmentation ready. "
                                    + "objects=%d runtime=%dms threshold=%.3f",
                            results.size(),
                            runtimeMs,
                            foregroundThreshold
                    )
            );

            return results;

        } finally {
            releaseContours(
                    contours
            );

            rgba.release();
            rgb.release();
            lab.release();
            gray.release();

            if (borderMask != null) {
                borderMask.release();
            }

            validPaperMask.release();
            sampleMask.release();
            lChannel.release();

            if (backgroundLab != null) {
                backgroundLab.release();
            }

            labDifference.release();
            grayDifference.release();
            foregroundScore.release();
            blurredScore.release();

            otsuMask.release();
            thresholdMask.release();
            edgeMask.release();
            combinedMask.release();
            cleanedMask.release();

            if (closeKernel != null) {
                closeKernel.release();
            }

            if (openKernel != null) {
                openKernel.release();
            }

            if (edgeKernel != null) {
                edgeKernel.release();
            }
        }
    }

    /**
     * A teljes bináris maszkot gyógyszerenként külön instance-maskokra
     * bontja.
     */
    private List<Mat> separateMedicineInstances(
            Mat sourceRgb,
            Mat binaryMask,
            double minimumArea
    ) {
        List<Mat> instances =
                new ArrayList<>();

        Mat componentLabels = new Mat();
        Mat componentStats = new Mat();
        Mat componentCentroids = new Mat();
        Mat componentMask = new Mat();

        try {
            int componentCount =
                    Imgproc.connectedComponentsWithStats(
                            binaryMask,
                            componentLabels,
                            componentStats,
                            componentCentroids
                    );

            for (int componentLabel = 1;
                 componentLabel < componentCount;
                 componentLabel++) {
                double[] areaValue =
                        componentStats.get(
                                componentLabel,
                                Imgproc.CC_STAT_AREA
                        );

                double componentArea =
                        areaValue == null
                                || areaValue.length == 0
                                ? 0.0
                                : areaValue[0];

                if (componentArea < minimumArea) {
                    continue;
                }

                Core.compare(
                        componentLabels,
                        new Scalar(componentLabel),
                        componentMask,
                        Core.CMP_EQ
                );

                List<Mat> splitMasks =
                        splitComponentWithWatershed(
                                sourceRgb,
                                componentMask,
                                minimumArea
                        );

                if (splitMasks.isEmpty()) {
                    instances.add(
                            componentMask.clone()
                    );
                } else {
                    instances.addAll(
                            splitMasks
                    );
                }
            }

            return instances;

        } finally {
            componentLabels.release();
            componentStats.release();
            componentCentroids.release();
            componentMask.release();
        }
    }

    /**
     * Egy összefüggő folton belül distance transform magokat keres,
     * majd watershed segítségével külön gyógyszerekre bontja.
     */
    private List<Mat> splitComponentWithWatershed(
            Mat sourceRgb,
            Mat componentMask,
            double minimumArea
    ) {
        List<Mat> results =
                new ArrayList<>();

        Mat distance = new Mat();
        Mat markers =
                Mat.zeros(
                        componentMask.rows(),
                        componentMask.cols(),
                        CvType.CV_32SC1
                );

        Mat outsideMask = new Mat();
        Mat seedMask =
                Mat.zeros(
                        componentMask.rows(),
                        componentMask.cols(),
                        CvType.CV_8UC1
                );

        Mat instanceMask = new Mat();
        Mat resultCloseKernel = null;

        try {
            Imgproc.distanceTransform(
                    componentMask,
                    distance,
                    Imgproc.DIST_L2,
                    5
            );

            Core.MinMaxLocResult distanceRange =
                    Core.minMaxLoc(
                            distance
                    );

            if (distanceRange.maxVal
                    < WATERSHED_LOCAL_MAX_MIN_RADIUS_PX) {
                results.add(
                        componentMask.clone()
                );

                return results;
            }

            List<DistancePeak> peaks =
                    findLocalDistancePeaks(
                            distance,
                            componentMask
                    );

            Log.d(
                    TAG,
                    "Watershed local maxima: "
                            + peaks.size()
            );

            /*
             * Egyetlen lokális maximum valóban egyetlen gyógyszert jelent.
             */
            if (peaks.size() <= 1) {
                results.add(
                        componentMask.clone()
                );

                return results;
            }

            /*
             * A komponensen kívüli biztos háttér címkéje 1.
             * A komponens belseje egyelőre 0, vagyis watershed-ismeretlen.
             */
            Core.bitwise_not(
                    componentMask,
                    outsideMask
            );

            markers.setTo(
                    new Scalar(1),
                    outsideMask
            );

            int markerLabel = 2;

            for (DistancePeak peak : peaks) {
                seedMask.setTo(
                        new Scalar(0)
                );

                int seedRadius =
                        clamp(
                                (int) Math.round(
                                        peak.value * 0.28
                                ),
                                1,
                                6
                        );

                Imgproc.circle(
                        seedMask,
                        peak.point,
                        seedRadius,
                        new Scalar(255),
                        -1
                );

                Core.bitwise_and(
                        seedMask,
                        componentMask,
                        seedMask
                );

                markers.setTo(
                        new Scalar(markerLabel),
                        seedMask
                );

                markerLabel++;
            }

            /*
             * Minden marker saját medencét kap. A markerhatár -1,
             * a komponensen kívüli háttér 1 marad.
             */
            Imgproc.watershed(
                    sourceRgb,
                    markers
            );

            resultCloseKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    WATERSHED_RESULT_CLOSE_SIZE,
                                    WATERSHED_RESULT_CLOSE_SIZE
                            )
                    );

            double minimumInstanceArea =
                    Math.max(
                            WATERSHED_MIN_INSTANCE_AREA_PX,
                            minimumArea
                                    * WATERSHED_MIN_INSTANCE_AREA_RATIO
                    );

            for (int peakIndex = 0;
                 peakIndex < peaks.size();
                 peakIndex++) {
                int label =
                        peakIndex + 2;

                Core.compare(
                        markers,
                        new Scalar(label),
                        instanceMask,
                        Core.CMP_EQ
                );

                Core.bitwise_and(
                        instanceMask,
                        componentMask,
                        instanceMask
                );

                Imgproc.morphologyEx(
                        instanceMask,
                        instanceMask,
                        Imgproc.MORPH_CLOSE,
                        resultCloseKernel,
                        new Point(-1, -1),
                        1
                );

                fillLargestExternalContour(
                        instanceMask
                );

                int instanceArea =
                        Core.countNonZero(
                                instanceMask
                        );

                if (instanceArea
                        >= minimumInstanceArea) {
                    results.add(
                            instanceMask.clone()
                    );
                }
            }

            /*
             * Fontos: több marker esetén többé nem állunk vissza az egész
             * nagy componentMask-ra. Az okozta korábban a két óriási
             * csoportmaszkot.
             */
            if (results.isEmpty()) {
                Log.w(
                        TAG,
                        "A lokális maximumok megvoltak, de a watershed "
                                + "nem adott használható régiót. "
                                + "Biztonsági fallback következik."
                );

                results.add(
                        componentMask.clone()
                );
            }

            Log.d(
                    TAG,
                    "Watershed instances: peaks="
                            + peaks.size()
                            + " accepted="
                            + results.size()
            );

            return results;

        } catch (Exception error) {
            Log.w(
                    TAG,
                    "A lokálismaximum-alapú watershed "
                            + "szétválasztás sikertelen.",
                    error
            );

            for (Mat result : results) {
                result.release();
            }

            results.clear();
            results.add(
                    componentMask.clone()
            );

            return results;

        } finally {
            if (resultCloseKernel != null) {
                resultCloseKernel.release();
            }

            distance.release();
            markers.release();
            outsideMask.release();
            seedMask.release();
            instanceMask.release();
        }
    }

    /**
     * A distance transform lokális maximumait keresi.
     *
     * A maximumszűrő ablaka a gyógyszer becsült sugarából származik,
     * ezért a közeli gyógyszerek külön maximumot kapnak, egy hosszúkás
     * kapszula összefüggő belső gerince viszont nem osztódik automatikusan
     * sok apró gyógyszerré.
     */
    private List<DistancePeak> findLocalDistancePeaks(
            Mat distance,
            Mat componentMask
    ) {
        List<DistancePeak> rawPeaks =
                new ArrayList<>();

        Mat dilatedDistance = new Mat();
        Mat dilatedMinusEpsilon = new Mat();
        Mat localMaximumMask = new Mat();
        Mat highDistanceFloat = new Mat();
        Mat highDistanceMask = new Mat();
        Mat peakLabels = new Mat();
        Mat peakStats = new Mat();
        Mat peakCentroids = new Mat();
        Mat peakComponentMask = new Mat();

        Mat localMaximumKernel = null;

        try {
            Core.MinMaxLocResult range =
                    Core.minMaxLoc(
                            distance
                    );

            double minimumPeakValue =
                    Math.max(
                            2.0,
                            range.maxVal
                                    * WATERSHED_LOCAL_MAX_VALUE_RATIO
                    );

            int localMaximumRadius =
                    clamp(
                            (int) Math.round(
                                    range.maxVal
                                            * WATERSHED_LOCAL_MAX_WINDOW_RATIO
                            ),
                            WATERSHED_LOCAL_MAX_MIN_RADIUS_PX,
                            WATERSHED_LOCAL_MAX_MAX_RADIUS_PX
                    );

            int kernelSize =
                    localMaximumRadius
                            * 2
                            + 1;

            localMaximumKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    kernelSize,
                                    kernelSize
                            )
                    );

            Imgproc.dilate(
                    distance,
                    dilatedDistance,
                    localMaximumKernel
            );

            /*
             * Lebegőpontos egyenlőség helyett kis epsilonnal hasonlítunk.
             */
            Core.subtract(
                    dilatedDistance,
                    new Scalar(0.0001),
                    dilatedMinusEpsilon
            );

            Core.compare(
                    distance,
                    dilatedMinusEpsilon,
                    localMaximumMask,
                    Core.CMP_GE
            );

            Imgproc.threshold(
                    distance,
                    highDistanceFloat,
                    minimumPeakValue,
                    255.0,
                    Imgproc.THRESH_BINARY
            );

            highDistanceFloat.convertTo(
                    highDistanceMask,
                    CvType.CV_8UC1
            );

            Core.bitwise_and(
                    localMaximumMask,
                    highDistanceMask,
                    localMaximumMask
            );

            Core.bitwise_and(
                    localMaximumMask,
                    componentMask,
                    localMaximumMask
            );

            int peakComponentCount =
                    Imgproc.connectedComponentsWithStats(
                            localMaximumMask,
                            peakLabels,
                            peakStats,
                            peakCentroids
                    );

            for (int label = 1;
                 label < peakComponentCount;
                 label++) {
                Core.compare(
                        peakLabels,
                        new Scalar(label),
                        peakComponentMask,
                        Core.CMP_EQ
                );

                Core.MinMaxLocResult peakValue =
                        Core.minMaxLoc(
                                distance,
                                peakComponentMask
                        );

                if (peakValue.maxVal
                        < minimumPeakValue
                        || peakValue.maxLoc == null) {
                    continue;
                }

                rawPeaks.add(
                        new DistancePeak(
                                new Point(
                                        Math.rint(
                                                peakValue.maxLoc.x
                                        ),
                                        Math.rint(
                                                peakValue.maxLoc.y
                                        )
                                ),
                                peakValue.maxVal
                        )
                );
            }

            /*
             * Nagyobb sugarú maximumok előre, azonos értéknél stabil
             * felülről lefelé és balról jobbra sorrend.
             */
            Collections.sort(
                    rawPeaks,
                    Comparator
                            .comparingDouble(
                                    (DistancePeak peak) ->
                                            -peak.value
                            )
                            .thenComparingDouble(
                                    peak ->
                                            peak.point.y
                            )
                            .thenComparingDouble(
                                    peak ->
                                            peak.point.x
                            )
            );

            List<DistancePeak> filtered =
                    new ArrayList<>();

            for (DistancePeak candidate : rawPeaks) {
                boolean sameMedicine =
                        false;

                for (DistancePeak accepted : filtered) {
                    double centerDistance =
                            Math.hypot(
                                    candidate.point.x
                                            - accepted.point.x,
                                    candidate.point.y
                                            - accepted.point.y
                            );

                    double referenceRadius =
                            Math.min(
                                    candidate.value,
                                    accepted.value
                            );

                    if (centerDistance
                            < Math.max(
                            2.0,
                            referenceRadius
                                    * WATERSHED_NEAR_PEAK_RATIO
                    )) {
                        sameMedicine = true;
                        break;
                    }

                    if (hasContinuousDistanceRidge(
                            distance,
                            candidate,
                            accepted
                    )) {
                        sameMedicine = true;
                        break;
                    }
                }

                if (!sameMedicine) {
                    filtered.add(
                            candidate
                    );
                }

                if (filtered.size()
                        >= WATERSHED_MAX_INSTANCE_COUNT) {
                    break;
                }
            }

            /*
             * A markerazonosítók stabil térbeli sorrendet kapnak.
             */
            Collections.sort(
                    filtered,
                    Comparator
                            .comparingDouble(
                                    (DistancePeak peak) ->
                                            peak.point.y
                            )
                            .thenComparingDouble(
                                    peak ->
                                            peak.point.x
                            )
            );

            return filtered;

        } finally {
            if (localMaximumKernel != null) {
                localMaximumKernel.release();
            }

            dilatedDistance.release();
            dilatedMinusEpsilon.release();
            localMaximumMask.release();
            highDistanceFloat.release();
            highDistanceMask.release();
            peakLabels.release();
            peakStats.release();
            peakCentroids.release();
            peakComponentMask.release();
        }
    }

    /**
     * Ha két maximum között végig magas distance-transform gerinc fut,
     * akkor valószínűleg ugyanannak a hosszúkás gyógyszernek a két
     * pontjáról van szó.
     *
     * Két külön, egymáshoz érő tabletta között a distance-érték a
     * kontaktusnál jelentősen leesik, ezért külön marker maradnak.
     */
    private boolean hasContinuousDistanceRidge(
            Mat distance,
            DistancePeak first,
            DistancePeak second
    ) {
        double pointDistance =
                Math.hypot(
                        first.point.x
                                - second.point.x,
                        first.point.y
                                - second.point.y
                );

        int sampleCount =
                Math.max(
                        8,
                        (int) Math.ceil(
                                pointDistance
                        )
                );

        double minimumValue =
                Double.POSITIVE_INFINITY;

        for (int sample = 0;
             sample <= sampleCount;
             sample++) {
            double ratio =
                    sample
                            / (double) sampleCount;

            int x =
                    clamp(
                            (int) Math.round(
                                    first.point.x
                                            + (
                                            second.point.x
                                                    - first.point.x
                                    ) * ratio
                            ),
                            0,
                            distance.cols() - 1
                    );

            int y =
                    clamp(
                            (int) Math.round(
                                    first.point.y
                                            + (
                                            second.point.y
                                                    - first.point.y
                                    ) * ratio
                            ),
                            0,
                            distance.rows() - 1
                    );

            double[] value =
                    distance.get(
                            y,
                            x
                    );

            double currentValue =
                    value == null
                            || value.length == 0
                            ? 0.0
                            : value[0];

            minimumValue =
                    Math.min(
                            minimumValue,
                            currentValue
                    );
        }

        double referenceValue =
                Math.min(
                        first.value,
                        second.value
                );

        return minimumValue
                >= referenceValue
                * WATERSHED_RIDGE_MERGE_RATIO;
    }


    private SegmentCandidate createCandidate(
            MatOfPoint originalContour,
            int imageWidth,
            int imageHeight,
            double minimumArea,
            double maximumArea,
            Mat sourceEdgeMask,
            Mat sourceRgb,
            Mat separatedInstanceMask
    ) throws Exception {
        if (originalContour == null
                || originalContour.empty()) {
            return null;
        }

        MatOfInt hullIndexes =
                new MatOfInt();

        MatOfPoint hullContour =
                new MatOfPoint();

        Mat individualMask =
                Mat.zeros(
                        imageHeight,
                        imageWidth,
                        CvType.CV_8UC1
                );

        try {
            Imgproc.convexHull(
                    originalContour,
                    hullIndexes
            );

            Point[] sourcePoints =
                    originalContour.toArray();

            int[] indexes =
                    hullIndexes.toArray();

            List<Point> hullPoints =
                    new ArrayList<>();

            for (int index : indexes) {
                if (index >= 0
                        && index < sourcePoints.length) {
                    hullPoints.add(
                            sourcePoints[index]
                    );
                }
            }

            if (hullPoints.size() < 3) {
                return null;
            }

            hullContour.fromList(
                    hullPoints
            );

            double hullArea =
                    Imgproc.contourArea(
                            hullContour
                    );

            if (hullArea < minimumArea
                    || hullArea > maximumArea) {
                return null;
            }

            Rect rect =
                    Imgproc.boundingRect(
                            hullContour
                    );

            if (rect.width
                    < MIN_COMPONENT_SIDE_PX
                    || rect.height
                    < MIN_COMPONENT_SIDE_PX) {
                return null;
            }

            if (touchesRejectedBorder(
                    rect,
                    imageWidth,
                    imageHeight
            )) {
                return null;
            }

            /*
             * Axis-aligned arány: általános zajszűréshez marad.
             */
            double aspectRatio =
                    Math.max(
                            rect.width,
                            rect.height
                    )
                            / (double) Math.max(
                            1,
                            Math.min(
                                    rect.width,
                                    rect.height
                            )
                    );

            if (aspectRatio > MAX_ASPECT_RATIO) {
                return null;
            }

            /*
             * Elforgatásfüggetlen oldalarány. Ez akadályozza meg, hogy
             * egy ferdén fekvő hosszúkás gyógyszert kör alakúnak
             * minősítsünk.
             */
            double orientedAspectRatio =
                    calculateOrientedAspectRatio(
                            hullContour
                    );

            double originalArea =
                    Math.max(
                            1.0,
                            Imgproc.contourArea(
                                    originalContour
                            )
                    );

            double solidity =
                    originalArea
                            / Math.max(
                            1.0,
                            hullArea
                    );

            if (solidity < MIN_SOLIDITY) {
                return null;
            }

            /*
             * A végleges feldolgozás kezdőmaszkja kizárólag az ehhez az
             * egy gyógyszerhez tartozó watershed-példány. Nem használjuk
             * a több gyógyszert összekötő teljes foltot.
             */
            if (separatedInstanceMask == null
                    || separatedInstanceMask.empty()) {
                return null;
            }

            separatedInstanceMask.copyTo(
                    individualMask
            );

            fillLargestExternalContour(
                    individualMask
            );

            /*
             * A világos tabletta széle néha szinte azonos a papír
             * színével. Ilyenkor a küszöbölés a körvonal egy részét
             * elveszítheti, a convex hull pedig egy átlós húrral zárja le
             * az alakot. Ez okozta a képen látható V alakú kivágást.
             *
             * Kerek vagy ovális, megfelelően tömör objektumnál ezért
             * determinisztikus ellipszisillesztéssel egészítjük ki a
             * hiányzó körvonalat. A művelet szigorú geometriai
             * ellenőrzésekhez kötött, így nem fut le tetszőleges alakokra.
             */
            /*
             * A korábbi kör/ellipszis kapcsolgatást teljesen kiváltjuk.
             *
             * A durva convex hull kizárólag kezdőterület. A végleges
             * alakot a forráskép közeli valódi éleire illesztett
             * elforgatott ellipszis adja. Így:
             *
             * - a kör alakú tabletta közel kör alakú ellipszist kap;
             * - a hosszúkás gyógyszer hosszúkás ellipszist kap;
             * - a V alakú egyenes lezárás nem kerülhet a végleges maszkba.
             */
            String completionMode =
                    applyEdgeFittedEllipseMask(
                            individualMask,
                            rect,
                            hullArea,
                            maximumArea,
                            sourceEdgeMask,
                            sourceRgb
                    );

            /*
             * A GrabCut/ellipszis nem léphet át a szomszéd gyógyszer
             * watershed-területére.
             */
            constrainMaskToInstanceOwner(
                    individualMask,
                    separatedInstanceMask,
                    rect
            );

            /*
             * Egyetlen fix zárás betömi a kapszulán/tablettán belüli
             * törővonal vagy felirat okozta apró lyukakat.
             */
            Mat stabilizingKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(5, 5)
                    );

            try {
                Imgproc.morphologyEx(
                        individualMask,
                        individualMask,
                        Imgproc.MORPH_CLOSE,
                        stabilizingKernel
                );
            } finally {
                stabilizingKernel.release();
            }

            /*
             * A zárás után még egyszer érvényesítjük a saját
             * instance-határt.
             */
            constrainMaskToInstanceOwner(
                    individualMask,
                    separatedInstanceMask,
                    rect
            );

            List<MatOfPoint> finalContours =
                    new ArrayList<>();

            Mat maskCopy =
                    individualMask.clone();

            Mat finalHierarchy =
                    new Mat();

            try {
                Imgproc.findContours(
                        maskCopy,
                        finalContours,
                        finalHierarchy,
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_SIMPLE
                );
            } finally {
                maskCopy.release();
                finalHierarchy.release();
            }

            MatOfPoint finalContour =
                    largestContour(
                            finalContours
                    );

            if (finalContour == null) {
                releaseContours(
                        finalContours
                );
                return null;
            }

            List<Point> simplifiedPolygon =
                    simplifyContour(
                            finalContour
                    );

            if (simplifiedPolygon.size() < 3) {
                releaseContours(
                        finalContours
                );
                return null;
            }

            /*
             * A hash pontosan abból a poligonmaszkból készül, amelyet a
             * PillScanFragment is újrarajzol az 05_binary_mask.png fájlhoz.
             */
            Mat stablePolygonMask =
                    Mat.zeros(
                            imageHeight,
                            imageWidth,
                            CvType.CV_8UC1
                    );

            MatOfPoint stablePolygonContour =
                    new MatOfPoint();

            Rect finalRect;
            double finalArea;
            String hash;

            try {
                stablePolygonContour.fromList(
                        simplifiedPolygon
                );

                Imgproc.fillPoly(
                        stablePolygonMask,
                        Collections.singletonList(
                                stablePolygonContour
                        ),
                        new Scalar(255)
                );

                finalRect =
                        Imgproc.boundingRect(
                                stablePolygonContour
                        );

                finalArea =
                        Imgproc.contourArea(
                                stablePolygonContour
                        );

                hash =
                        sha256Mask(
                                stablePolygonMask
                        );

            } finally {
                stablePolygonContour.release();
                stablePolygonMask.release();
            }

            releaseContours(
                    finalContours
            );

            double centerX =
                    finalRect.x
                            + finalRect.width / 2.0;

            double centerY =
                    finalRect.y
                            + finalRect.height / 2.0;

            return new SegmentCandidate(
                    simplifiedPolygon,
                    new PixelBox(
                            finalRect.x,
                            finalRect.y,
                            finalRect.x
                                    + finalRect.width,
                            finalRect.y
                                    + finalRect.height
                    ),
                    finalArea,
                    solidity,
                    orientedAspectRatio,
                    centerX,
                    centerY,
                    hash,
                    completionMode
            );

        } finally {
            hullIndexes.release();
            hullContour.release();
            individualMask.release();
        }
    }

    /**
     * Kerek vagy ovális tabletta hiányzó szektorának determinisztikus
     * kiegészítése.
     *
     * A módszer nem minden kontúrra rajzol ellipszist. Csak akkor egészít:
     *
     * - ha legalább 5 kontúrpont áll rendelkezésre;
     * - ha az alak nem túl hosszúkás;
     * - ha a kontúr tömörsége megfelelő;
     * - ha az ellipszis középpontja a meglévő objektum közelében marad;
     * - ha az ellipszis tengelyei és területe nem nőnek túl nagyra;
     * - ha az eredmény nem lépi túl a megengedett objektumterületet.
     *
     * Így a hiányos kerek tablettát kijavítja, de két összeérő
     * gyógyszert vagy szabálytalan zajfoltot nem próbál egyetlen nagy
     * ellipszissé alakítani.
     */
    /**
     * Kerek vagy ovális tabletta hiányzó szektorának determinisztikus
     * kiegészítése.
     *
     * Első lépésként a közel kör alakú objektumokra a legkisebb
     * befoglaló kört használjuk. Ez a meglévő körívből akkor is képes
     * visszaállítani a teljes tablettát, ha a küszöbölés egy nagy
     * körcikket elveszített.
     *
     * Ha az objektum nem eléggé kör alakú, akkor a korábbi, szigorúan
     * ellenőrzött ellipszisillesztés következik.
     */
    /**
     * A durva maszk közelében található valódi képi élekre
     * elforgatott ellipszist illeszt, majd ezt használja végleges
     * gyógyszermaszkként.
     *
     * Ez a megközelítés nem próbálja a már hibás, V alakban lezárt
     * kontúrt körrel vagy ellipszissel utólag toldozni. Ehelyett
     * közvetlenül a forráskép éleiből építi újra a teljes alakot.
     */
    /**
     * Szoros, elcsúszásmentes végleges maszk készítése.
     *
     * 1. A közeli élekből készül egy kezdeti elforgatott ellipszis.
     * 2. Ez csak a GrabCut inicializálásához szolgál.
     * 3. A GrabCut a forráskép tényleges színeloszlása alapján választja
     *    szét a gyógyszert a papírtól és az árnyéktól.
     * 4. A kiválasztott előtér-komponensre új ellipszis készül.
     * 5. Kizárólag ez az új ellipszis lesz a végleges maszk.
     *
     * A hibás convex hull soha nem kerül vissza a végső maszkba, ezért
     * nem maradhat V alakú lezárás.
     */
    private String applyEdgeFittedEllipseMask(
            Mat individualMask,
            Rect coarseRect,
            double coarseArea,
            double maximumArea,
            Mat sourceEdgeMask,
            Mat sourceRgb
    ) {
        if (individualMask == null
                || individualMask.empty()
                || sourceEdgeMask == null
                || sourceEdgeMask.empty()
                || sourceRgb == null
                || sourceRgb.empty()
                || coarseRect == null
                || coarseRect.width <= 0
                || coarseRect.height <= 0) {
            return applyEllipseFallbackFromMask(
                    individualMask,
                    "tight_ellipse_fallback"
            );
        }

        Mat influenceMask = new Mat();
        Mat nearbyEdges = new Mat();
        MatOfPoint edgePoints = new MatOfPoint();
        MatOfPoint2f fitPoints = new MatOfPoint2f();

        Mat influenceKernel = null;

        try {
            int coarseMajor = Math.max(
                    coarseRect.width,
                    coarseRect.height
            );

            int influenceSize =
                    clamp(
                            (int) Math.round(
                                    coarseMajor
                                            * EDGE_INFLUENCE_RADIUS_RATIO
                            ),
                            5,
                            13
                    );

            if (influenceSize % 2 == 0) {
                influenceSize++;
            }

            influenceKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    influenceSize,
                                    influenceSize
                            )
                    );

            Imgproc.dilate(
                    individualMask,
                    influenceMask,
                    influenceKernel
            );

            Core.bitwise_and(
                    sourceEdgeMask,
                    influenceMask,
                    nearbyEdges
            );

            Core.findNonZero(
                    nearbyEdges,
                    edgePoints
            );

            Point[] rawPoints =
                    edgePoints.toArray();

            RotatedRect initialEllipse =
                    createInitialEllipse(
                            rawPoints,
                            individualMask
                    );

            if (!isFiniteEllipse(
                    initialEllipse
            )) {
                return applyEllipseFallbackFromMask(
                        individualMask,
                        "tight_ellipse_fallback"
                );
            }

            RotatedRect grabCutEllipse =
                    refineEllipseWithGrabCut(
                            sourceRgb,
                            individualMask,
                            initialEllipse,
                            maximumArea
                    );

            RotatedRect finalEllipse =
                    isFiniteEllipse(
                            grabCutEllipse
                    )
                            ? grabCutEllipse
                            : initialEllipse;

            double finalWidth =
                    finalEllipse.size.width
                            * GRABCUT_FINAL_ELLIPSE_SCALE;

            double finalHeight =
                    finalEllipse.size.height
                            * GRABCUT_FINAL_ELLIPSE_SCALE;

            double finalArea =
                    Math.PI
                            * finalWidth
                            * finalHeight
                            / 4.0;

            if (finalWidth <= 1.0
                    || finalHeight <= 1.0
                    || finalArea > maximumArea * 1.08) {
                return applyEllipseFallbackFromMask(
                        individualMask,
                        "tight_ellipse_fallback"
                );
            }

            Mat finalMask =
                    Mat.zeros(
                            individualMask.rows(),
                            individualMask.cols(),
                            CvType.CV_8UC1
                    );

            try {
                Imgproc.ellipse(
                        finalMask,
                        finalEllipse.center,
                        new Size(
                                finalWidth / 2.0,
                                finalHeight / 2.0
                        ),
                        finalEllipse.angle,
                        0.0,
                        360.0,
                        new Scalar(255),
                        -1
                );

                /*
                 * Nem uniózzuk a durva maszkkal. Ez a lényegi javítás:
                 * az árnyék, a háttér és a V alakú hull-rész nem maradhat
                 * benne a végleges maszkon.
                 */
                finalMask.copyTo(
                        individualMask
                );

            } finally {
                finalMask.release();
            }

            Log.d(
                    TAG,
                    String.format(
                            Locale.ROOT,
                            "Tight mask accepted: "
                                    + "center=(%.2f, %.2f) "
                                    + "size=(%.2f, %.2f) "
                                    + "angle=%.2f mode=%s",
                            finalEllipse.center.x,
                            finalEllipse.center.y,
                            finalWidth,
                            finalHeight,
                            finalEllipse.angle,
                            isFiniteEllipse(grabCutEllipse)
                                    ? "grabcut_tight_ellipse"
                                    : "edge_ellipse_fallback"
                    )
            );

            return isFiniteEllipse(
                    grabCutEllipse
            )
                    ? "grabcut_tight_ellipse"
                    : "edge_ellipse_fallback";

        } catch (Exception error) {
            Log.e(
                    TAG,
                    "A szoros GrabCut-maszk készítése sikertelen.",
                    error
            );

            return applyEllipseFallbackFromMask(
                    individualMask,
                    "tight_ellipse_fallback"
            );

        } finally {
            if (influenceKernel != null) {
                influenceKernel.release();
            }

            influenceMask.release();
            nearbyEdges.release();
            edgePoints.release();
            fitPoints.release();
        }
    }

    /**
     * Kezdeti ellipszis a közeli élekből. Ha az élek nem elegendők,
     * közvetlenül a durva maszk kontúrjára illesztünk ellipszist.
     */
    private RotatedRect createInitialEllipse(
            Point[] rawPoints,
            Mat coarseMask
    ) {
        MatOfPoint2f points =
                new MatOfPoint2f();

        try {
            if (rawPoints != null
                    && rawPoints.length
                    >= MIN_EDGE_POINTS_FOR_ELLIPSE) {
                points.fromArray(
                        rawPoints
                );

                RotatedRect first =
                        Imgproc.fitEllipse(
                                points
                        );

                List<Point> boundaryPoints =
                        selectEllipseBoundaryPoints(
                                rawPoints,
                                first
                        );

                if (boundaryPoints.size()
                        >= MIN_EDGE_POINTS_FOR_ELLIPSE) {
                    points.fromList(
                            boundaryPoints
                    );

                    RotatedRect refined =
                            Imgproc.fitEllipse(
                                    points
                            );

                    if (isFiniteEllipse(
                            refined
                    )) {
                        return refined;
                    }
                }

                if (isFiniteEllipse(
                        first
                )) {
                    return first;
                }
            }

            List<MatOfPoint> contours =
                    new ArrayList<>();

            Mat copy =
                    coarseMask.clone();

            Mat hierarchy =
                    new Mat();

            try {
                Imgproc.findContours(
                        copy,
                        contours,
                        hierarchy,
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_NONE
                );
            } finally {
                copy.release();
                hierarchy.release();
            }

            MatOfPoint largest =
                    largestContour(
                            contours
                    );

            if (largest == null) {
                releaseContours(
                        contours
                );
                return null;
            }

            Point[] contourPoints =
                    largest.toArray();

            RotatedRect result = null;

            if (contourPoints.length >= 5) {
                points.fromArray(
                        contourPoints
                );

                result =
                        Imgproc.fitEllipse(
                                points
                        );
            } else {
                points.fromArray(
                        contourPoints
                );

                result =
                        Imgproc.minAreaRect(
                                points
                        );
            }

            releaseContours(
                    contours
            );

            return result;

        } finally {
            points.release();
        }
    }

    /**
     * GrabCut inicializálása a kezdeti ellipszisből, majd a tényleges
     * előtérkomponensre szoros ellipszis illesztése.
     */
    private RotatedRect refineEllipseWithGrabCut(
            Mat sourceRgb,
            Mat coarseMask,
            RotatedRect initialEllipse,
            double maximumArea
    ) {
        Mat grabMask =
                new Mat(
                        sourceRgb.rows(),
                        sourceRgb.cols(),
                        CvType.CV_8UC1,
                        new Scalar(
                                Imgproc.GC_BGD
                        )
                );

        Mat roiMask =
                Mat.zeros(
                        sourceRgb.rows(),
                        sourceRgb.cols(),
                        CvType.CV_8UC1
                );

        Mat probableForegroundMask =
                Mat.zeros(
                        sourceRgb.rows(),
                        sourceRgb.cols(),
                        CvType.CV_8UC1
                );

        Mat allowedForegroundMask =
                new Mat();

        Mat definiteForegroundMask =
                Mat.zeros(
                        sourceRgb.rows(),
                        sourceRgb.cols(),
                        CvType.CV_8UC1
                );

        Mat erodedCoarse =
                new Mat();

        Mat foregroundOne =
                new Mat();

        Mat foregroundThree =
                new Mat();

        Mat foregroundMask =
                new Mat();

        Mat cleanedForeground =
                new Mat();

        Mat selectedComponent =
                new Mat();

        Mat backgroundModel =
                new Mat();

        Mat foregroundModel =
                new Mat();

        Mat erosionKernel = null;
        Mat closeKernel = null;
        Mat allowedKernel = null;

        try {
            drawScaledEllipse(
                    roiMask,
                    initialEllipse,
                    GRABCUT_ROI_SCALE,
                    255
            );

            drawScaledEllipse(
                    probableForegroundMask,
                    initialEllipse,
                    GRABCUT_PROBABLE_FOREGROUND_SCALE,
                    255
            );

            drawScaledEllipse(
                    definiteForegroundMask,
                    initialEllipse,
                    GRABCUT_DEFINITE_FOREGROUND_SCALE,
                    255
            );

            /*
             * A valószínű előtér nem nyúlhat át a szomszédos
             * gyógyszerre. Az adott watershed-példányt enyhén kitágítjuk,
             * és a probable foreground ellipszist erre korlátozzuk.
             */
            int allowedSize =
                    clamp(
                            (int) Math.round(
                                    Math.min(
                                            initialEllipse.size.width,
                                            initialEllipse.size.height
                                    ) * 0.10
                            ),
                            3,
                            9
                    );

            if (allowedSize % 2 == 0) {
                allowedSize++;
            }

            allowedKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    allowedSize,
                                    allowedSize
                            )
                    );

            Imgproc.dilate(
                    coarseMask,
                    allowedForegroundMask,
                    allowedKernel
            );

            Core.bitwise_and(
                    probableForegroundMask,
                    allowedForegroundMask,
                    probableForegroundMask
            );

            /*
             * A nagy ROI csak valószínű háttér. Az inicializáló ellipszis
             * belseje valószínű előtér, a belső mag biztos előtér.
             */
            grabMask.setTo(
                    new Scalar(
                            Imgproc.GC_PR_BGD
                    ),
                    roiMask
            );

            grabMask.setTo(
                    new Scalar(
                            Imgproc.GC_PR_FGD
                    ),
                    probableForegroundMask
            );

            grabMask.setTo(
                    new Scalar(
                            Imgproc.GC_FGD
                    ),
                    definiteForegroundMask
            );

            /*
             * A durva maszk erodált belseje további biztos előtér.
             * A széleket nem használjuk biztos előtérként, mert ott lehet
             * az árnyék vagy a korábbi V alakú hiba.
             */
            int erosionSize =
                    Math.max(
                            3,
                            (int) Math.round(
                                    Math.min(
                                            initialEllipse.size.width,
                                            initialEllipse.size.height
                                    ) * 0.13
                            )
                    );

            if (erosionSize % 2 == 0) {
                erosionSize++;
            }

            erosionKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    erosionSize,
                                    erosionSize
                            )
                    );

            Imgproc.erode(
                    coarseMask,
                    erodedCoarse,
                    erosionKernel
            );

            grabMask.setTo(
                    new Scalar(
                            Imgproc.GC_FGD
                    ),
                    erodedCoarse
            );

            /*
             * Fix RNG-seed + fix iterációszám: azonos bemenethez azonos
             * GrabCut-eredmény.
             */
            Core.setRNGSeed(
                    20260729
            );

            Imgproc.grabCut(
                    sourceRgb,
                    grabMask,
                    new Rect(),
                    backgroundModel,
                    foregroundModel,
                    GRABCUT_ITERATIONS,
                    Imgproc.GC_INIT_WITH_MASK
            );

            Core.compare(
                    grabMask,
                    new Scalar(
                            Imgproc.GC_FGD
                    ),
                    foregroundOne,
                    Core.CMP_EQ
            );

            Core.compare(
                    grabMask,
                    new Scalar(
                            Imgproc.GC_PR_FGD
                    ),
                    foregroundThree,
                    Core.CMP_EQ
            );

            Core.bitwise_or(
                    foregroundOne,
                    foregroundThree,
                    foregroundMask
            );

            closeKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(3, 3)
                    );

            Imgproc.morphologyEx(
                    foregroundMask,
                    cleanedForeground,
                    Imgproc.MORPH_CLOSE,
                    closeKernel,
                    new Point(-1, -1),
                    1
            );

            if (!selectForegroundComponent(
                    cleanedForeground,
                    definiteForegroundMask,
                    selectedComponent
            )) {
                return null;
            }

            /*
             * A GrabCut eredménye még tartalmazhatja a gyógyszerhez
             * kapcsolódó vetett árnyékot. A következő lépés a biztos
             * gyógyszermag fényességéből indul ki, eltávolítja a sötét
             * külső árnyékot, majd az esetleges belső törővonal-lyukakat
             * visszatölti.
             */
            suppressAttachedShadow(
                    sourceRgb,
                    selectedComponent,
                    definiteForegroundMask,
                    initialEllipse
            );

            List<MatOfPoint> contours =
                    new ArrayList<>();

            Mat contourInput =
                    selectedComponent.clone();

            Mat hierarchy =
                    new Mat();

            try {
                Imgproc.findContours(
                        contourInput,
                        contours,
                        hierarchy,
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_NONE
                );
            } finally {
                contourInput.release();
                hierarchy.release();
            }

            MatOfPoint contour =
                    largestContour(
                            contours
                    );

            if (contour == null
                    || contour.toArray().length < 5) {
                releaseContours(
                        contours
                );
                return null;
            }

            MatOfPoint2f contourPoints =
                    new MatOfPoint2f(
                            contour.toArray()
                    );

            RotatedRect result;

            try {
                result =
                        Imgproc.fitEllipse(
                                contourPoints
                        );
            } finally {
                contourPoints.release();
            }

            double selectedArea =
                    Core.countNonZero(
                            selectedComponent
                    );

            double ellipseArea =
                    Math.PI
                            * result.size.width
                            * result.size.height
                            / 4.0;

            double initialMinor =
                    Math.min(
                            initialEllipse.size.width,
                            initialEllipse.size.height
                    );

            double initialMajor =
                    Math.max(
                            initialEllipse.size.width,
                            initialEllipse.size.height
                    );

            double resultMinor =
                    Math.min(
                            result.size.width,
                            result.size.height
                    );

            double resultMajor =
                    Math.max(
                            result.size.width,
                            result.size.height
                    );

            double centerShift =
                    Math.hypot(
                            result.center.x
                                    - initialEllipse.center.x,
                            result.center.y
                                    - initialEllipse.center.y
                    );

            double maximumCenterShift =
                    Math.max(
                            5.0,
                            initialMajor
                                    * GRABCUT_MAX_CENTER_SHIFT_RATIO
                    );

            Core.bitwise_and(
                    selectedComponent,
                    definiteForegroundMask,
                    foregroundOne
            );

            double coreArea =
                    Math.max(
                            1.0,
                            Core.countNonZero(
                                    definiteForegroundMask
                            )
                    );

            double coreCoverage =
                    Core.countNonZero(
                            foregroundOne
                    )
                            / coreArea;

            releaseContours(
                    contours
            );

            if (!isFiniteEllipse(
                    result
            )
                    || resultMinor
                    < initialMinor
                    * GRABCUT_MIN_AXIS_RATIO_TO_INITIAL
                    || resultMajor
                    > initialMajor
                    * GRABCUT_MAX_AXIS_RATIO_TO_INITIAL
                    || centerShift
                    > maximumCenterShift
                    || coreCoverage
                    < GRABCUT_MIN_CORE_COVERAGE
                    || ellipseArea
                    > maximumArea * 1.06
                    || selectedArea <= 0.0) {
                return null;
            }

            return result;

        } catch (Exception error) {
            Log.w(
                    TAG,
                    "A GrabCut finomítás kihagyásra került.",
                    error
            );

            return null;

        } finally {
            if (erosionKernel != null) {
                erosionKernel.release();
            }

            if (closeKernel != null) {
                closeKernel.release();
            }

            if (allowedKernel != null) {
                allowedKernel.release();
            }

            grabMask.release();
            roiMask.release();
            probableForegroundMask.release();
            allowedForegroundMask.release();
            definiteForegroundMask.release();
            erodedCoarse.release();
            foregroundOne.release();
            foregroundThree.release();
            foregroundMask.release();
            cleanedForeground.release();
            selectedComponent.release();
            backgroundModel.release();
            foregroundModel.release();
        }
    }

    /**
     * Eltávolítja a gyógyszerhez kapcsolódó sötét vetett árnyékot.
     *
     * A művelet kizárólag a GrabCut által már kiválasztott komponensen
     * belül dolgozik, ezért a világos papírháttér nem kerülhet vissza.
     *
     * A belső score-vonal vagy felirat átmenetileg kieshet a
     * fényességküszöb miatt, de a végén a legnagyobb külső kontúrt
     * kitöltjük, így ezek nem vágnak lyukat a gyógyszermaszkba.
     */
    private void suppressAttachedShadow(
            Mat sourceRgb,
            Mat selectedComponent,
            Mat definiteForegroundMask,
            RotatedRect initialEllipse
    ) {
        if (sourceRgb == null
                || sourceRgb.empty()
                || selectedComponent == null
                || selectedComponent.empty()
                || definiteForegroundMask == null
                || definiteForegroundMask.empty()
                || !isFiniteEllipse(initialEllipse)) {
            return;
        }

        Mat lab = new Mat();
        Mat lChannel = new Mat();
        Mat brightEnough = new Mat();
        Mat shadowSuppressed = new Mat();
        Mat refinedComponent = new Mat();
        Mat closeKernel = null;

        List<Mat> labChannels =
                new ArrayList<>();

        try {
            Imgproc.cvtColor(
                    sourceRgb,
                    lab,
                    Imgproc.COLOR_RGB2Lab
            );

            Core.split(
                    lab,
                    labChannels
            );

            if (labChannels.isEmpty()) {
                return;
            }

            labChannels.get(0).copyTo(
                    lChannel
            );

            /*
             * A gyógyszer biztos belső magjának átlagos fényessége.
             * A score-vonal csak kis terület, ezért az átlagot nem
             * befolyásolja jelentősen.
             */
            Scalar coreLightness =
                    Core.mean(
                            lChannel,
                            definiteForegroundMask
                    );

            double lightnessDrop =
                    clamp(
                            coreLightness.val[0]
                                    * SHADOW_L_DROP_RATIO,
                            SHADOW_MIN_L_DROP,
                            SHADOW_MAX_L_DROP
                    );

            double minimumPillLightness =
                    Math.max(
                            0.0,
                            coreLightness.val[0]
                                    - lightnessDrop
                    );

            Imgproc.threshold(
                    lChannel,
                    brightEnough,
                    minimumPillLightness,
                    255.0,
                    Imgproc.THRESH_BINARY
            );

            /*
             * Csak a már kiválasztott GrabCut-komponensen belül
             * távolítjuk el a sötét pixeleket. Így a világos papír nem
             * kerülhet hozzá a gyógyszerhez.
             */
            Core.bitwise_and(
                    selectedComponent,
                    brightEnough,
                    shadowSuppressed
            );

            /*
             * A biztos belső gyógyszermagot minden esetben megtartjuk.
             */
            Core.bitwise_or(
                    shadowSuppressed,
                    definiteForegroundMask,
                    shadowSuppressed
            );

            int kernelSize =
                    (int) Math.round(
                            Math.min(
                                    initialEllipse.size.width,
                                    initialEllipse.size.height
                            )
                                    * SHADOW_CLOSE_KERNEL_RATIO
                    );

            kernelSize =
                    Math.max(
                            SHADOW_MIN_CLOSE_KERNEL,
                            Math.min(
                                    SHADOW_MAX_CLOSE_KERNEL,
                                    kernelSize
                            )
                    );

            if (kernelSize % 2 == 0) {
                kernelSize++;
            }

            closeKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    kernelSize,
                                    kernelSize
                            )
                    );

            /*
             * A belső felirat, törővonal és apró sötét pontok által
             * létrehozott réseket bezárjuk.
             */
            Imgproc.morphologyEx(
                    shadowSuppressed,
                    shadowSuppressed,
                    Imgproc.MORPH_CLOSE,
                    closeKernel,
                    new Point(-1, -1),
                    2
            );

            if (!selectForegroundComponent(
                    shadowSuppressed,
                    definiteForegroundMask,
                    refinedComponent
            )) {
                return;
            }

            /*
             * A kiválasztott külső kontúrt teljesen kitöltjük. Ettől a
             * score-vonal nem válik maszkon belüli lyukká.
             */
            fillLargestExternalContour(
                    refinedComponent
            );

            if (Core.countNonZero(
                    refinedComponent
            ) > 0) {
                refinedComponent.copyTo(
                        selectedComponent
                );
            }

            Log.d(
                    TAG,
                    String.format(
                            Locale.ROOT,
                            "Shadow suppression: "
                                    + "coreL=%.2f minL=%.2f "
                                    + "pixelsBefore=%d pixelsAfter=%d",
                            coreLightness.val[0],
                            minimumPillLightness,
                            Core.countNonZero(
                                    shadowSuppressed
                            ),
                            Core.countNonZero(
                                    selectedComponent
                            )
                    )
            );

        } catch (Exception error) {
            /*
             * Az árnyékeltávolítás finomító lépés. Hiba esetén a
             * korábbi GrabCut-komponens marad, a teljes felismerés nem áll
             * le.
             */
            Log.w(
                    TAG,
                    "Az árnyékeltávolítás kihagyásra került.",
                    error
            );

        } finally {
            for (Mat channel : labChannels) {
                if (channel != null) {
                    channel.release();
                }
            }

            if (closeKernel != null) {
                closeKernel.release();
            }

            lab.release();
            lChannel.release();
            brightEnough.release();
            shadowSuppressed.release();
            refinedComponent.release();
        }
    }

    /**
     * A kész gyógyszermaszkot a saját watershed-példányához korlátozza.
     *
     * Kis margót engedünk, hogy a watershed határán levő valódi
     * gyógyszerszél ne vesszen el, de a szomszéd gyógyszer már ne
     * kerülhessen bele.
     */
    private void constrainMaskToInstanceOwner(
            Mat finalMask,
            Mat instanceOwnerMask,
            Rect ownerRect
    ) {
        if (finalMask == null
                || finalMask.empty()
                || instanceOwnerMask == null
                || instanceOwnerMask.empty()
                || ownerRect == null) {
            return;
        }

        Mat allowedOwner = new Mat();
        Mat ownerKernel = null;

        try {
            int marginRadius =
                    clamp(
                            (int) Math.round(
                                    Math.min(
                                            ownerRect.width,
                                            ownerRect.height
                                    )
                                            * INSTANCE_OWNER_MARGIN_RATIO
                            ),
                            INSTANCE_OWNER_MIN_MARGIN_PX,
                            INSTANCE_OWNER_MAX_MARGIN_PX
                    );

            int kernelSize =
                    marginRadius
                            * 2
                            + 1;

            ownerKernel =
                    Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            new Size(
                                    kernelSize,
                                    kernelSize
                            )
                    );

            Imgproc.dilate(
                    instanceOwnerMask,
                    allowedOwner,
                    ownerKernel
            );

            Core.bitwise_and(
                    finalMask,
                    allowedOwner,
                    finalMask
            );

            fillLargestExternalContour(
                    finalMask
            );

        } finally {
            if (ownerKernel != null) {
                ownerKernel.release();
            }

            allowedOwner.release();
        }
    }

    /**
     * A maszk legnagyobb külső kontúrját kitöltve tartja meg.
     */
    private void fillLargestExternalContour(
            Mat mask
    ) {
        if (mask == null
                || mask.empty()) {
            return;
        }

        List<MatOfPoint> contours =
                new ArrayList<>();

        Mat contourInput =
                mask.clone();

        Mat hierarchy =
                new Mat();

        try {
            Imgproc.findContours(
                    contourInput,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            MatOfPoint largest =
                    largestContour(
                            contours
                    );

            if (largest == null) {
                return;
            }

            mask.setTo(
                    new Scalar(0)
            );

            List<MatOfPoint> singleContour =
                    new ArrayList<>();

            singleContour.add(
                    largest
            );

            Imgproc.drawContours(
                    mask,
                    singleContour,
                    0,
                    new Scalar(255),
                    -1
            );

        } finally {
            contourInput.release();
            hierarchy.release();
            releaseContours(
                    contours
            );
        }
    }

    private void drawScaledEllipse(
            Mat target,
            RotatedRect ellipse,
            double scale,
            int value
    ) {
        Imgproc.ellipse(
                target,
                ellipse.center,
                new Size(
                        Math.max(
                                1.0,
                                ellipse.size.width
                                        * scale
                                        / 2.0
                        ),
                        Math.max(
                                1.0,
                                ellipse.size.height
                                        * scale
                                        / 2.0
                        )
                ),
                ellipse.angle,
                0.0,
                360.0,
                new Scalar(value),
                -1
        );
    }

    /**
     * A GrabCut komponensei közül azt választja, amely a biztos
     * előtérmaggal a legjobban átfed.
     */
    private boolean selectForegroundComponent(
            Mat foreground,
            Mat definiteForeground,
            Mat output
    ) {
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        Mat overlap = new Mat();
        Mat componentMask = new Mat();

        try {
            int count =
                    Imgproc.connectedComponentsWithStats(
                            foreground,
                            labels,
                            stats,
                            centroids
                    );

            int bestLabel = -1;
            int bestOverlap = 0;
            int bestArea = 0;

            for (int label = 1;
                 label < count;
                 label++) {
                Core.compare(
                        labels,
                        new Scalar(label),
                        componentMask,
                        Core.CMP_EQ
                );

                Core.bitwise_and(
                        componentMask,
                        definiteForeground,
                        overlap
                );

                int overlapPixels =
                        Core.countNonZero(
                                overlap
                        );

                int area =
                        (int) stats.get(
                                label,
                                Imgproc.CC_STAT_AREA
                        )[0];

                if (overlapPixels > bestOverlap
                        || (overlapPixels == bestOverlap
                        && area > bestArea)) {
                    bestLabel = label;
                    bestOverlap = overlapPixels;
                    bestArea = area;
                }
            }

            if (bestLabel < 0
                    || bestOverlap <= 0) {
                return false;
            }

            Core.compare(
                    labels,
                    new Scalar(bestLabel),
                    output,
                    Core.CMP_EQ
            );

            return true;

        } finally {
            labels.release();
            stats.release();
            centroids.release();
            overlap.release();
            componentMask.release();
        }
    }

    /**
     * Biztonsági tartalék: a durva maszkra ellipszist illeszt, és azt
     * használja végleges maszkként. A durva V alak soha nem marad meg.
     */
    private String applyEllipseFallbackFromMask(
            Mat mask,
            String mode
    ) {
        if (mask == null
                || mask.empty()) {
            return mode;
        }

        List<MatOfPoint> contours =
                new ArrayList<>();

        Mat copy =
                mask.clone();

        Mat hierarchy =
                new Mat();

        try {
            Imgproc.findContours(
                    copy,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
            );

            MatOfPoint contour =
                    largestContour(
                            contours
                    );

            if (contour == null) {
                return mode;
            }

            Point[] points =
                    contour.toArray();

            MatOfPoint2f pointMatrix =
                    new MatOfPoint2f(
                            points
                    );

            Mat ellipseMask =
                    Mat.zeros(
                            mask.rows(),
                            mask.cols(),
                            CvType.CV_8UC1
                    );

            try {
                RotatedRect ellipse =
                        points.length >= 5
                                ? Imgproc.fitEllipse(
                                pointMatrix
                        )
                                : Imgproc.minAreaRect(
                                pointMatrix
                        );

                if (!isFiniteEllipse(
                        ellipse
                )) {
                    return mode;
                }

                Imgproc.ellipse(
                        ellipseMask,
                        ellipse.center,
                        new Size(
                                Math.max(
                                        1.0,
                                        ellipse.size.width / 2.0
                                ),
                                Math.max(
                                        1.0,
                                        ellipse.size.height / 2.0
                                )
                        ),
                        ellipse.angle,
                        0.0,
                        360.0,
                        new Scalar(255),
                        -1
                );

                ellipseMask.copyTo(
                        mask
                );

                return mode;

            } finally {
                pointMatrix.release();
                ellipseMask.release();
            }

        } finally {
            copy.release();
            hierarchy.release();
            releaseContours(
                    contours
            );
        }
    }

    /**
     * Az első ellipszis belső törővonalhoz tartozó pontjait és a túl
     * távoli árnyékéleket eltávolítja.
     */
    private List<Point> selectEllipseBoundaryPoints(
            Point[] points,
            RotatedRect ellipse
    ) {
        List<Point> result =
                new ArrayList<>();

        if (points == null
                || points.length == 0
                || !isFiniteEllipse(
                ellipse
        )) {
            return result;
        }

        double halfWidth =
                ellipse.size.width / 2.0;

        double halfHeight =
                ellipse.size.height / 2.0;

        if (halfWidth <= 0.0
                || halfHeight <= 0.0) {
            return result;
        }

        double radians =
                Math.toRadians(
                        ellipse.angle
                );

        double cos =
                Math.cos(
                        radians
                );

        double sin =
                Math.sin(
                        radians
                );

        for (Point point : points) {
            if (point == null
                    || !Double.isFinite(
                    point.x
            )
                    || !Double.isFinite(
                    point.y
            )) {
                continue;
            }

            double dx =
                    point.x
                            - ellipse.center.x;

            double dy =
                    point.y
                            - ellipse.center.y;

            double rotatedX =
                    cos * dx
                            + sin * dy;

            double rotatedY =
                    -sin * dx
                            + cos * dy;

            double normalizedRadius =
                    Math.sqrt(
                            rotatedX
                                    * rotatedX
                                    / (halfWidth
                                    * halfWidth)
                                    + rotatedY
                                    * rotatedY
                                    / (halfHeight
                                    * halfHeight)
                    );

            if (normalizedRadius
                    >= MIN_BOUNDARY_NORMALIZED_RADIUS
                    && normalizedRadius
                    <= MAX_BOUNDARY_NORMALIZED_RADIUS) {
                result.add(
                        point
                );
            }
        }

        return result;
    }

    private boolean isFiniteEllipse(
            RotatedRect ellipse
    ) {
        return ellipse != null
                && ellipse.center != null
                && ellipse.size != null
                && Double.isFinite(
                ellipse.center.x
        )
                && Double.isFinite(
                ellipse.center.y
        )
                && Double.isFinite(
                ellipse.size.width
        )
                && Double.isFinite(
                ellipse.size.height
        )
                && Double.isFinite(
                ellipse.angle
        )
                && ellipse.size.width > 0.0
                && ellipse.size.height > 0.0;
    }

    private String completeTabletLikeShape(
            Mat individualMask,
            MatOfPoint hullContour,
            Rect originalRect,
            double hullArea,
            double maximumArea,
            double orientedAspectRatio,
            double solidity,
            Mat sourceEdgeMask
    ) {
        if (individualMask == null
                || individualMask.empty()
                || hullContour == null
                || hullContour.empty()) {
            return "none";
        }

        Point[] hullPoints =
                hullContour.toArray();

        if (hullPoints.length < 3) {
            return "none";
        }

        /*
         * 1. KÖR ALAKÚ TABLETTA JAVÍTÁSA
         *
         * A minEnclosingCircle nem a hiányos körcikk területét próbálja
         * követni. A megmaradt külső körív pontjait foglalja a legkisebb
         * körbe, ezért a V alakban hiányzó rész is visszakerül.
         */
        String circleCompletion =
                completeRoundTabletWithEnclosingCircle(
                        individualMask,
                        hullPoints,
                        originalRect,
                        hullArea,
                        maximumArea,
                        orientedAspectRatio,
                        solidity,
                        sourceEdgeMask
                );

        if (!"none".equals(
                circleCompletion
        )) {
            return circleCompletion;
        }

        /*
         * 2. OVÁLIS TABLETTA JAVÍTÁSA
         */
        if (hullPoints.length < 5) {
            return "none";
        }

        if (orientedAspectRatio
                > MAX_COMPLETION_ASPECT_RATIO) {
            return "none";
        }

        if (solidity
                < MIN_COMPLETION_SOLIDITY) {
            return "none";
        }

        MatOfPoint2f ellipsePoints =
                new MatOfPoint2f(
                        hullPoints
                );

        Mat ellipseMask =
                Mat.zeros(
                        individualMask.rows(),
                        individualMask.cols(),
                        CvType.CV_8UC1
                );

        try {
            RotatedRect ellipse =
                    Imgproc.fitEllipse(
                            ellipsePoints
                    );

            double ellipseWidth =
                    ellipse.size.width;

            double ellipseHeight =
                    ellipse.size.height;

            if (!Double.isFinite(
                    ellipse.center.x
            )
                    || !Double.isFinite(
                    ellipse.center.y
            )
                    || !Double.isFinite(
                    ellipseWidth
            )
                    || !Double.isFinite(
                    ellipseHeight
            )
                    || ellipseWidth <= 0.0
                    || ellipseHeight <= 0.0) {
                return "none";
            }

            double originalCenterX =
                    originalRect.x
                            + originalRect.width / 2.0;

            double originalCenterY =
                    originalRect.y
                            + originalRect.height / 2.0;

            double maximumOffsetX =
                    Math.max(
                            4.0,
                            originalRect.width
                                    * MAX_ELLIPSE_CENTER_OFFSET_RATIO
                    );

            double maximumOffsetY =
                    Math.max(
                            4.0,
                            originalRect.height
                                    * MAX_ELLIPSE_CENTER_OFFSET_RATIO
                    );

            if (Math.abs(
                    ellipse.center.x
                            - originalCenterX
            ) > maximumOffsetX
                    || Math.abs(
                    ellipse.center.y
                            - originalCenterY
            ) > maximumOffsetY) {
                return "none";
            }

            double ellipseMinor =
                    Math.min(
                            ellipseWidth,
                            ellipseHeight
                    );

            double ellipseMajor =
                    Math.max(
                            ellipseWidth,
                            ellipseHeight
                    );

            double rectMinor =
                    Math.min(
                            originalRect.width,
                            originalRect.height
                    );

            double rectMajor =
                    Math.max(
                            originalRect.width,
                            originalRect.height
                    );

            if (ellipseMinor
                    > rectMinor
                    * MAX_ELLIPSE_AXIS_GROWTH
                    || ellipseMajor
                    > rectMajor
                    * MAX_ELLIPSE_AXIS_GROWTH) {
                return "none";
            }

            double ellipseArea =
                    Math.PI
                            * ellipseWidth
                            * ellipseHeight
                            / 4.0;

            double areaRatio =
                    ellipseArea
                            / Math.max(
                            1.0,
                            hullArea
                    );

            if (areaRatio
                    < MIN_ELLIPSE_TO_HULL_AREA_RATIO
                    || areaRatio
                    > MAX_ELLIPSE_TO_HULL_AREA_RATIO
                    || ellipseArea
                    > maximumArea * 1.05) {
                return "none";
            }

            Imgproc.ellipse(
                    ellipseMask,
                    ellipse.center,
                    new Size(
                            ellipseWidth / 2.0,
                            ellipseHeight / 2.0
                    ),
                    ellipse.angle,
                    0.0,
                    360.0,
                    new Scalar(255),
                    -1
            );

            Core.bitwise_or(
                    individualMask,
                    ellipseMask,
                    individualMask
            );

            return "fit_ellipse_union";

        } catch (Exception ellipseError) {
            Log.w(
                    TAG,
                    "Az ellipszis alapú alak-kiegészítés "
                            + "kihagyásra került.",
                    ellipseError
            );

            return "none";

        } finally {
            ellipsePoints.release();
            ellipseMask.release();
        }
    }

    /**
     * Közel kör alakú tablettánál a megmaradt külső pontokra meghatározza
     * a legkisebb befoglaló kört, majd annak maszkját egyesíti az eredeti
     * maszkkal.
     */
    private String completeRoundTabletWithEnclosingCircle(
            Mat individualMask,
            Point[] hullPoints,
            Rect originalRect,
            double hullArea,
            double maximumArea,
            double orientedAspectRatio,
            double solidity,
            Mat sourceEdgeMask
    ) {
        if (orientedAspectRatio
                > MAX_CIRCLE_COMPLETION_ORIENTED_ASPECT_RATIO) {
            return "none";
        }

        if (solidity
                < MIN_CIRCLE_COMPLETION_SOLIDITY) {
            return "none";
        }

        MatOfPoint2f points =
                new MatOfPoint2f(
                        hullPoints
                );

        Mat circleMask =
                Mat.zeros(
                        individualMask.rows(),
                        individualMask.cols(),
                        CvType.CV_8UC1
                );

        try {
            Point center =
                    new Point();

            float[] radiusOutput =
                    new float[1];

            Imgproc.minEnclosingCircle(
                    points,
                    center,
                    radiusOutput
            );

            double radius =
                    radiusOutput[0];

            if (!Double.isFinite(
                    center.x
            )
                    || !Double.isFinite(
                    center.y
            )
                    || !Double.isFinite(
                    radius
            )
                    || radius <= 1.0) {
                return "none";
            }

            double originalCenterX =
                    originalRect.x
                            + originalRect.width / 2.0;

            double originalCenterY =
                    originalRect.y
                            + originalRect.height / 2.0;

            /*
             * A hiányzó körcikk miatt a bounding rect közepe kissé
             * eltolódhat, ezért itt megengedőbb ellenőrzést használunk.
             */
            double maximumCenterOffset =
                    Math.max(
                            8.0,
                            Math.max(
                                    originalRect.width,
                                    originalRect.height
                            ) * 0.42
                    );

            double centerOffset =
                    Math.hypot(
                            center.x - originalCenterX,
                            center.y - originalCenterY
                    );

            if (centerOffset
                    > maximumCenterOffset) {
                return "none";
            }

            double circleDiameter =
                    radius * 2.0;

            double rectMajor =
                    Math.max(
                            originalRect.width,
                            originalRect.height
                    );

            if (circleDiameter
                    > rectMajor
                    * MAX_CIRCLE_DIAMETER_TO_RECT_RATIO) {
                return "none";
            }

            double circleArea =
                    Math.PI
                            * radius
                            * radius;

            double circleToHullRatio =
                    circleArea
                            / Math.max(
                            1.0,
                            hullArea
                    );

            if (circleToHullRatio
                    < MIN_CIRCLE_TO_HULL_AREA_RATIO
                    || circleToHullRatio
                    > MAX_CIRCLE_TO_HULL_AREA_RATIO
                    || circleArea
                    > maximumArea * 1.10) {
                return "none";
            }

            /*
             * Enyhe, fix 3,5%-os ráhagyás biztosítja, hogy az
             * antialiasolt világos külső pixelsor se maradjon le.
             */
            /*
             * A Canny-élek a nagyon világos tabletta és a fehér papír
             * találkozásánál gyengék lehetnek. Ez okozta, hogy az előző
             * verzió a valóban kör alakú tablettánál is letiltotta a
             * körkiegészítést.
             *
             * Az él-támogatást ezért csak diagnosztikai célból számoljuk.
             * A körmaszk alkalmazásáról az elforgatásfüggetlen oldalarány,
             * a tömörség, a terület és a középpont ellenőrzése dönt.
             */
            CircleEdgeSupport edgeSupport =
                    calculateCircleEdgeSupport(
                            sourceEdgeMask,
                            center,
                            radius
                    );

            Log.d(
                    TAG,
                    String.format(
                            Locale.ROOT,
                            "Round completion accepted: "
                                    + "orientedAspect=%.3f "
                                    + "edgeSupport=%.3f "
                                    + "quadrants=%d",
                            orientedAspectRatio,
                            edgeSupport.supportRatio,
                            edgeSupport.supportedQuadrants
                    )
            );

            int completedRadius =
                    Math.max(
                            1,
                            (int) Math.ceil(
                                    radius
                                            * CIRCLE_COMPLETION_RADIUS_PADDING
                            )
                    );

            Imgproc.circle(
                    circleMask,
                    center,
                    completedRadius,
                    new Scalar(255),
                    -1
            );

            Core.bitwise_or(
                    individualMask,
                    circleMask,
                    individualMask
            );

            return "min_enclosing_circle_union";

        } catch (Exception circleError) {
            Log.w(
                    TAG,
                    "A befoglaló kör alapú alak-kiegészítés "
                            + "kihagyásra került.",
                    circleError
            );

            return "none";

        } finally {
            points.release();
            circleMask.release();
        }
    }

    /**
     * A kontúr minimális elforgatott befoglaló téglalapjából számított
     * oldalarány. Ellentétben a boundingRect értékével, a gyógyszer
     * forgatási szögétől gyakorlatilag független.
     */
    private double calculateOrientedAspectRatio(
            MatOfPoint contour
    ) {
        if (contour == null
                || contour.empty()
                || contour.rows() < 3) {
            return Double.POSITIVE_INFINITY;
        }

        MatOfPoint2f points =
                new MatOfPoint2f(
                        contour.toArray()
                );

        try {
            RotatedRect rotatedRect =
                    Imgproc.minAreaRect(
                            points
                    );

            double width =
                    rotatedRect.size.width;

            double height =
                    rotatedRect.size.height;

            double minor =
                    Math.min(
                            width,
                            height
                    );

            double major =
                    Math.max(
                            width,
                            height
                    );

            if (!Double.isFinite(minor)
                    || !Double.isFinite(major)
                    || minor <= 0.0) {
                return Double.POSITIVE_INFINITY;
            }

            return major / minor;

        } finally {
            points.release();
        }
    }

    /**
     * Megvizsgálja, hogy a tervezett kör kerületén mennyi valódi Canny-él
     * található. A már dilatált edgeMask miatt egy kis radiális tolerancia
     * elegendő.
     */
    private CircleEdgeSupport calculateCircleEdgeSupport(
            Mat edgeMask,
            Point center,
            double radius
    ) {
        if (edgeMask == null
                || edgeMask.empty()
                || center == null
                || radius <= 1.0) {
            return new CircleEdgeSupport(
                    0.0,
                    0
            );
        }

        int radialTolerance =
                Math.max(
                        2,
                        (int) Math.round(
                                radius * 0.055
                        )
                );

        int supportedSamples = 0;

        int[] quadrantSamples =
                new int[4];

        int[] quadrantSupported =
                new int[4];

        for (int sample = 0;
             sample < CIRCLE_EDGE_SAMPLE_COUNT;
             sample++) {
            double angle =
                    2.0
                            * Math.PI
                            * sample
                            / CIRCLE_EDGE_SAMPLE_COUNT;

            int quadrant =
                    Math.min(
                            3,
                            sample
                                    * 4
                                    / CIRCLE_EDGE_SAMPLE_COUNT
                    );

            quadrantSamples[quadrant]++;

            boolean supported = false;

            for (int radialOffset =
                 -radialTolerance;
                 radialOffset <= radialTolerance;
                 radialOffset++) {
                double sampledRadius =
                        radius + radialOffset;

                int x =
                        (int) Math.round(
                                center.x
                                        + Math.cos(angle)
                                        * sampledRadius
                        );

                int y =
                        (int) Math.round(
                                center.y
                                        + Math.sin(angle)
                                        * sampledRadius
                        );

                if (x < 0
                        || y < 0
                        || x >= edgeMask.cols()
                        || y >= edgeMask.rows()) {
                    continue;
                }

                double[] pixel =
                        edgeMask.get(
                                y,
                                x
                        );

                if (pixel != null
                        && pixel.length > 0
                        && pixel[0] > 0.0) {
                    supported = true;
                    break;
                }
            }

            if (supported) {
                supportedSamples++;
                quadrantSupported[quadrant]++;
            }
        }

        int supportedQuadrants = 0;

        for (int quadrant = 0;
             quadrant < 4;
             quadrant++) {
            double quadrantRatio =
                    quadrantSamples[quadrant] == 0
                            ? 0.0
                            : quadrantSupported[quadrant]
                              / (double) quadrantSamples[quadrant];

            /*
             * Egy hiányos körtablettánál legfeljebb egy kvadráns lehet
             * gyengébb. A többi háromban legalább mérsékelt él-támogatást
             * várunk el.
             */
            if (quadrantRatio >= 0.20) {
                supportedQuadrants++;
            }
        }

        return new CircleEdgeSupport(
                supportedSamples
                        / (double) CIRCLE_EDGE_SAMPLE_COUNT,
                supportedQuadrants
        );
    }

    private Mat createBorderMask(
            int width,
            int height
    ) {
        Mat mask =
                Mat.zeros(
                        height,
                        width,
                        CvType.CV_8UC1
                );

        int border =
                Math.max(
                        4,
                        (int) Math.round(
                                Math.min(
                                        width,
                                        height
                                )
                                        * BORDER_RATIO
                        )
                );

        Imgproc.rectangle(
                mask,
                new Point(0, 0),
                new Point(
                        width - 1,
                        height - 1
                ),
                new Scalar(255),
                border
        );

        return mask;
    }

    private boolean touchesRejectedBorder(
            Rect rect,
            int imageWidth,
            int imageHeight
    ) {
        return rect.x <= BORDER_REJECTION_PX
                || rect.y <= BORDER_REJECTION_PX
                || rect.x + rect.width
                >= imageWidth - BORDER_REJECTION_PX
                || rect.y + rect.height
                >= imageHeight - BORDER_REJECTION_PX;
    }

    private List<Point> simplifyContour(
            MatOfPoint contour
    ) {
        MatOfPoint2f curve =
                new MatOfPoint2f(
                        contour.toArray()
                );

        MatOfPoint2f approximation =
                new MatOfPoint2f();

        try {
            double perimeter =
                    Imgproc.arcLength(
                            curve,
                            true
                    );

            double epsilon =
                    Math.max(
                            0.55,
                            perimeter * 0.0035
                    );

            Imgproc.approxPolyDP(
                    curve,
                    approximation,
                    epsilon,
                    true
            );

            Point[] points =
                    approximation.toArray();

            List<Point> result =
                    new ArrayList<>();

            for (Point point : points) {
                result.add(
                        new Point(
                                Math.rint(point.x),
                                Math.rint(point.y)
                        )
                );
            }

            return result;

        } finally {
            curve.release();
            approximation.release();
        }
    }

    private MatOfPoint largestContour(
            List<MatOfPoint> contours
    ) {
        MatOfPoint best = null;
        double bestArea = 0.0;

        for (MatOfPoint contour : contours) {
            double area =
                    Imgproc.contourArea(
                            contour
                    );

            if (area > bestArea) {
                bestArea = area;
                best = contour;
            }
        }

        return best;
    }

    private String sha256Mask(
            Mat mask
    ) throws Exception {
        if (mask == null
                || mask.empty()
                || mask.type()
                != CvType.CV_8UC1) {
            throw new Exception(
                    "A maszkhasonlításhoz érvénytelen Mat érkezett."
            );
        }

        int length =
                (int) (
                        mask.total()
                                * mask.channels()
                );

        byte[] bytes =
                new byte[length];

        mask.get(
                0,
                0,
                bytes
        );

        MessageDigest digest =
                MessageDigest.getInstance(
                        "SHA-256"
                );

        byte[] hash =
                digest.digest(
                        bytes
                );

        StringBuilder result =
                new StringBuilder(
                        hash.length * 2
                );

        for (byte value : hash) {
            result.append(
                    String.format(
                            Locale.ROOT,
                            "%02x",
                            value & 0xff
                    )
            );
        }

        return result.toString();
    }

    private void validateBitmap(
            Bitmap bitmap
    ) throws Exception {
        if (bitmap == null) {
            throw new Exception(
                    "A szegmentálandó Bitmap null."
            );
        }

        if (bitmap.isRecycled()) {
            throw new Exception(
                    "A szegmentálandó Bitmap már fel lett szabadítva."
            );
        }

        if (bitmap.getWidth() <= 0
                || bitmap.getHeight() <= 0) {
            throw new Exception(
                    "A szegmentálandó Bitmap mérete érvénytelen."
            );
        }
    }

    private void releaseContours(
            List<MatOfPoint> contours
    ) {
        if (contours == null) {
            return;
        }

        for (MatOfPoint contour : contours) {
            if (contour != null) {
                contour.release();
            }
        }

        contours.clear();
    }

    /**
     * Egész számok korlátozása.
     *
     * A watershed marker-, kernel- és koordinátaméretei int értékek.
     * Külön overload szükséges, mert a double clamp eredménye nem
     * alakítható automatikusan int-té.
     */
    private int clamp(
            int value,
            int minimum,
            int maximum
    ) {
        return Math.max(
                minimum,
                Math.min(
                        maximum,
                        value
                )
        );
    }

    private double clamp(
            double value,
            double minimum,
            double maximum
    ) {
        return Math.max(
                minimum,
                Math.min(
                        maximum,
                        value
                )
        );
    }

    private double round(
            double value
    ) {
        return Math.round(
                value * 1000.0
        ) / 1000.0;
    }

    private static final class DistancePeak {
        final Point point;
        final double value;

        DistancePeak(
                Point point,
                double value
        ) {
            this.point = point;
            this.value = value;
        }
    }

    private static final class CircleEdgeSupport {
        final double supportRatio;
        final int supportedQuadrants;

        CircleEdgeSupport(
                double supportRatio,
                int supportedQuadrants
        ) {
            this.supportRatio = supportRatio;
            this.supportedQuadrants = supportedQuadrants;
        }
    }

    private static final class SegmentCandidate {
        final List<Point> polygon;
        final PixelBox box;
        final double area;
        final double solidity;
        final double orientedAspectRatio;
        final double centerX;
        final double centerY;
        final String maskSha256;
        final String completionMode;

        SegmentCandidate(
                List<Point> polygon,
                PixelBox box,
                double area,
                double solidity,
                double orientedAspectRatio,
                double centerX,
                double centerY,
                String maskSha256,
                String completionMode
        ) {
            this.polygon = polygon;
            this.box = box;
            this.area = area;
            this.solidity = solidity;
            this.orientedAspectRatio = orientedAspectRatio;
            this.centerX = centerX;
            this.centerY = centerY;
            this.maskSha256 = maskSha256;
            this.completionMode =
                    completionMode == null
                            ? "none"
                            : completionMode;
        }
    }

    public static final class PixelBox {
        public final int left;
        public final int top;
        public final int right;
        public final int bottom;

        PixelBox(
                int left,
                int top,
                int right,
                int bottom
        ) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }

        public int width() {
            return Math.max(
                    0,
                    right - left
            );
        }

        public int height() {
            return Math.max(
                    0,
                    bottom - top
            );
        }
    }

    /**
     * A meglévő PillScanFragment számára a korábbi Gemini-eredménnyel
     * azonos felületet biztosítja, de az adatok teljesen helyi OpenCV
     * feldolgozásból származnak.
     */
    public static final class SegmentationResult {
        private final String model;
        private final String label;
        private final List<Point> polygonPixels;
        private final PixelBox pixelBox;
        private final String structuredJson;
        private final String diagnosticsJson;
        private final long runtimeMs;
        private final String maskSha256;

        SegmentationResult(
                String model,
                String label,
                List<Point> polygonPixels,
                PixelBox pixelBox,
                String structuredJson,
                String diagnosticsJson,
                long runtimeMs,
                String maskSha256
        ) {
            this.model = model;
            this.label = label;
            this.polygonPixels =
                    Collections.unmodifiableList(
                            new ArrayList<>(
                                    polygonPixels
                            )
                    );
            this.pixelBox = pixelBox;
            this.structuredJson = structuredJson;
            this.diagnosticsJson = diagnosticsJson;
            this.runtimeMs = runtimeMs;
            this.maskSha256 = maskSha256;
        }

        public String getModel() {
            return model;
        }

        public String getLabel() {
            return label;
        }

        public List<Point> getPolygonPixels() {
            return polygonPixels;
        }

        public PixelBox getPixelBox() {
            return pixelBox;
        }

        public String getStructuredJson() {
            return structuredJson;
        }

        /**
         * A metódus neve kompatibilitási okból maradt meg.
         * Nincs API-válasz: a determinisztikus diagnosztikai JSON-t adja.
         */
        public String getRawApiResponse() {
            return diagnosticsJson;
        }

        public long getRuntimeMs() {
            return runtimeMs;
        }

        public String getMaskSha256() {
            return maskSha256;
        }
    }
}