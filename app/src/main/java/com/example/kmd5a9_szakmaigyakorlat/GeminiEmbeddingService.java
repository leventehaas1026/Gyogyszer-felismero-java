package com.example.kmd5a9_szakmaigyakorlat;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Point;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * Gemini 3.6 Flash alapú gyógyszerszegmentálás a hivatalos Interactions API
 * formátumával, valamint Gemini Embedding 2 képi embedding készítés.
 *
 * A szegmentálási válasz dokumentált formátuma:
 * - box_2d: [ymin, xmin, ymax, xmax], a teljes képen 0..1000-re normalizálva
 * - mask: [x, y] pontok, a box_2d BELSEJÉBEN 0..1000-re normalizálva
 * - label: szöveges címke
 */
public final class GeminiEmbeddingService {

    private static final String TAG = "GeminiMedicine";

    private static final String API_KEY =
            "AQ.Ab8RN6Le2p2tHzZBS5_7GXvU9HQz_jdGHk5_XGJy7h9IR5s_2g";

    public static final String SEGMENTATION_MODEL =
            "gemini-3.6-flash";

    public static final String EMBEDDING_MODEL =
            "gemini-embedding-2";

    private static final String INTERACTIONS_URL =
            "https://generativelanguage.googleapis.com/v1beta/interactions";

    private static final String EMBEDDING_URL =
            "https://generativelanguage.googleapis.com/v1beta/models/"
                    + EMBEDDING_MODEL
                    + ":embedContent";

    private static final int SEGMENTATION_MAX_SIDE = 768;
    private static final int EMBEDDING_MAX_SIDE = 512;
    private static final int DEFAULT_EMBEDDING_DIMENSION = 512;
    private static final int MAX_SEGMENTED_MEDICINES = 8;


    private static final int API_TIMEOUT_SECONDS = 25;

    private final OkHttpClient segmentationClient =
            new OkHttpClient.Builder()
                    .connectTimeout(3, TimeUnit.SECONDS)
                    .writeTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .readTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .callTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .retryOnConnectionFailure(false)
                    .build();

    private final OkHttpClient embeddingClient =
            new OkHttpClient.Builder()
                    .connectTimeout(3, TimeUnit.SECONDS)
                    .writeTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .readTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .callTimeout(API_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .retryOnConnectionFailure(false)
                    .build();


    public SegmentationResult segmentMedicine(Bitmap bitmap) throws Exception {
        List<SegmentationResult> results = segmentMedicines(bitmap);

        return results.get(0);
    }


    public List<SegmentationResult> segmentMedicines(
            Bitmap bitmap
    ) throws Exception {
        validateBitmap(bitmap);
        validateApiKey();

        EncodedImage encodedImage = encodeBitmap(
                bitmap,
                Bitmap.CompressFormat.PNG,
                100,
                SEGMENTATION_MAX_SIDE,
                "image/png"
        );

        JSONObject requestBody = createInteractionRequest(encodedImage);

        long startedAt = SystemClock.elapsedRealtime();

        String rawApiResponse = postJson(
                segmentationClient,
                INTERACTIONS_URL,
                requestBody.toString(),
                "szegmentálás"
        );

        long runtimeMilliseconds =
                SystemClock.elapsedRealtime() - startedAt;

        double runtimeSeconds =
                millisecondsToSeconds(runtimeMilliseconds);

        String outputText = cleanJsonText(
                extractInteractionOutputText(rawApiResponse)
        );

        JSONObject root = new JSONObject(outputText);
        JSONArray boxes = root.optJSONArray("boxes");

        if (boxes == null || boxes.length() == 0) {
            throw new Exception(
                    "A Gemini nem adott vissza boxes tömböt. JSON: "
                            + shorten(outputText, 600)
            );
        }

        List<Candidate> candidates =
                chooseValidMedicineCandidates(
                        boxes,
                        bitmap.getWidth(),
                        bitmap.getHeight()
                );

        if (candidates.isEmpty()) {
            throw new Exception(
                    "A Gemini válaszában nincs használható gyógyszerszegmens."
            );
        }

        List<SegmentationResult> results =
                new ArrayList<>();

        for (Candidate candidate : candidates) {
            results.add(
                    new SegmentationResult(
                            candidate.label,
                            candidate.normalizedBox,
                            candidate.pixelBox,
                            candidate.polygonPixels,
                            outputText,
                            rawApiResponse,
                            runtimeSeconds,
                            SEGMENTATION_MODEL
                    )
            );
        }

        Log.d(
                TAG,
                "Szegmentálás kész: "
                        + formatSeconds(runtimeSeconds)
                        + " s; gyógyszerek="
                        + results.size()
        );

        return results;
    }

    public float[] createEmbedding(Bitmap bitmap) throws Exception {
        return createEmbedding(bitmap, DEFAULT_EMBEDDING_DIMENSION);
    }

    public float[] createEmbedding(
            Bitmap bitmap,
            int outputDimension
    ) throws Exception {
        validateBitmap(bitmap);
        validateApiKey();

        if (outputDimension < 128 || outputDimension > 3072) {
            throw new Exception(
                    "Az embedding dimenzió csak 128 és 3072 között lehet."
            );
        }

        EncodedImage encodedImage = encodeBitmap(
                bitmap,
                Bitmap.CompressFormat.PNG,
                100,
                EMBEDDING_MAX_SIDE,
                "image/png"
        );

        JSONArray parts = new JSONArray();
        parts.put(new JSONObject().put(
                "inline_data",
                new JSONObject()
                        .put("mime_type", encodedImage.mimeType)
                        .put("data", encodedImage.base64)
        ));

        JSONObject content = new JSONObject();
        content.put("parts", parts);

        JSONObject body = new JSONObject();
        body.put("content", content);
        body.put("output_dimensionality", outputDimension);

        long startedAt = SystemClock.elapsedRealtime();

        String rawResponse = postJson(
                embeddingClient,
                EMBEDDING_URL,
                body.toString(),
                "embedding"
        );

        long runtimeMilliseconds =
                SystemClock.elapsedRealtime() - startedAt;

        double runtimeSeconds =
                millisecondsToSeconds(runtimeMilliseconds);

        JSONObject responseObject = new JSONObject(rawResponse);

        if (responseObject.has("error")) {
            throw new Exception(
                    "Gemini embedding hiba: " + responseObject
            );
        }

        JSONArray values = findEmbeddingValues(responseObject);

        if (values == null || values.length() == 0) {
            throw new Exception(
                    "Nincs embedding.values a válaszban. Válasz: "
                            + shorten(rawResponse, 700)
            );
        }

        if (values.length() != outputDimension) {
            throw new Exception(
                    "Az embedding dimenzió eltér. Kért="
                            + outputDimension
                            + ", kapott="
                            + values.length()
            );
        }

        float[] result = new float[values.length()];

        for (int i = 0; i < values.length(); i++) {
            result[i] = (float) values.getDouble(i);
        }

        Log.d(
                TAG,
                "Embedding kész: "
                        + formatSeconds(runtimeSeconds)
                        + " s; dimenzió="
                        + result.length
        );

        return result;
    }

    private JSONObject createInteractionRequest(
            EncodedImage image
    ) throws Exception {
        String prompt =
                "Find and segment every distinct visible physical medicine object "
                        + "in the image, up to "
                        + MAX_SEGMENTED_MEDICINES
                        + " objects. A medicine object is a pill, tablet, caplet or "
                        + "capsule. Return one separate item in boxes for each medicine. "
                        + "Do not merge nearby or touching medicines into one segment, "
                        + "and do not return the same medicine more than once. Ignore "
                        + "paper, shadows, reflections, printed calibration elements, "
                        + "ArUco markers, dust and text. Each box_2d must tightly "
                        + "surround the whole visible medicine and use [ymin, xmin, "
                        + "ymax, xmax] normalized to 0-1000 over the full image. Each "
                        + "mask must trace the whole visible medicine, not only a "
                        + "highlight, imprint or shadow. Mask points are [x,y] "
                        + "coordinates normalized to 0-1000 inside its own box_2d. "
                        + "Use approximately 24 to 64 boundary points for every "
                        + "mask whenever the visible object size permits it. Order the "
                        + "points clockwise or counterclockwise and distribute them "
                        + "evenly around the external silhouette. Curved, circular and "
                        + "oval edges should use many closely spaced perimeter points. "
                        + "Do not approximate a round tablet with a triangle, diamond, "
                        + "pentagon, hexagon or another coarse polygon. Every returned "
                        + "point must lie on the outer medicine boundary. Do not return "
                        + "interior points, self-crossing point order, dents caused by "
                        + "shadows, or background around the medicine.";

        JSONArray input = new JSONArray();
        input.put(new JSONObject()
                .put("type", "text")
                .put("text", prompt));
        input.put(new JSONObject()
                .put("type", "image")
                .put("mime_type", image.mimeType)
                .put("data", image.base64));

        JSONObject responseFormat = new JSONObject();
        responseFormat.put("type", "text");
        responseFormat.put("mime_type", "application/json");
        responseFormat.put("schema", createSegmentationSchema());

        JSONObject generationConfig = new JSONObject();
        generationConfig.put("thinking_level", "minimal");
        generationConfig.put("temperature", 0.0);

        JSONObject request = new JSONObject();
        request.put("model", SEGMENTATION_MODEL);
        request.put("input", input);
        request.put("response_format", responseFormat);
        request.put("generation_config", generationConfig);

        return request;
    }

    private JSONObject createSegmentationSchema() throws Exception {
        JSONObject integer = new JSONObject()
                .put("type", "integer")
                .put("minimum", 0)
                .put("maximum", 1000);

        JSONObject coordinatePair = new JSONObject()
                .put("type", "array")
                .put("items", integer)
                .put("minItems", 2)
                .put("maxItems", 2);

        JSONObject boxArray = new JSONObject()
                .put("type", "array")
                .put("items", integer)
                .put("minItems", 4)
                .put("maxItems", 4);

        /*
         * A szerkezetet szándékosan egyszerűen tartjuk.
         *
         * A Gemini Interactions API ugyan támogat minItems/maxItems
         * mezőket, de a többszörösen egymásba ágyazott, sok korlátozást
         * tartalmazó séma egyes kéréseknél HTTP 400 invalid_request hibát
         * okozott. A pontszámra vonatkozó igényt ezért a prompt írja elő,
         * a kliens pedig külön ellenőrzi és stabilizálja a kontúrt.
         */
        JSONObject maskArray = new JSONObject()
                .put("type", "array")
                .put("items", coordinatePair)
                .put("minItems", 3);

        JSONObject item = new JSONObject()
                .put("type", "object")
                .put("properties", new JSONObject()
                        .put("box_2d", boxArray)
                        .put("mask", maskArray)
                        .put("label", new JSONObject()
                                .put("type", "string")))
                .put("required", new JSONArray()
                        .put("box_2d")
                        .put("mask")
                        .put("label"));

        JSONObject boxes = new JSONObject()
                .put("type", "array")
                .put("items", item)
                .put("minItems", 1)
                .put("maxItems", MAX_SEGMENTED_MEDICINES);

        return new JSONObject()
                .put("type", "object")
                .put("properties", new JSONObject()
                        .put("boxes", boxes))
                .put("required", new JSONArray().put("boxes"));
    }

    private List<Candidate> chooseValidMedicineCandidates(
            JSONArray boxes,
            int imageWidth,
            int imageHeight
    ) throws Exception {
        List<Candidate> candidates = new ArrayList<>();
        double imageArea = (double) imageWidth * imageHeight;

        for (int i = 0; i < boxes.length(); i++) {
            JSONObject item = boxes.optJSONObject(i);
            if (item == null) {
                continue;
            }

            JSONArray boxArray = item.optJSONArray("box_2d");
            JSONArray maskArray = item.optJSONArray("mask");
            String label = item.optString("label", "medicine");

            if (boxArray == null || boxArray.length() != 4
                    || maskArray == null || maskArray.length() < 3) {
                continue;
            }

            int[] normalizedBox = new int[]{
                    clampInt(boxArray.optInt(0), 0, 1000),
                    clampInt(boxArray.optInt(1), 0, 1000),
                    clampInt(boxArray.optInt(2), 0, 1000),
                    clampInt(boxArray.optInt(3), 0, 1000)
            };

            if (normalizedBox[2] <= normalizedBox[0]
                    || normalizedBox[3] <= normalizedBox[1]) {
                continue;
            }

            PixelBox pixelBox = normalizedBoxToPixelBox(
                    normalizedBox,
                    imageWidth,
                    imageHeight
            );

            List<Point> polygon = convertBoxLocalMaskToPixels(
                    normalizedBox,
                    maskArray,
                    imageWidth,
                    imageHeight
            );

            if (polygon.size() < 3) {
                continue;
            }

            double polygonArea = polygonArea(polygon);
            double boxArea = pixelBox.area();

            if (polygonArea < Math.max(12.0, imageArea * 0.00002)
                    || polygonArea > imageArea * 0.75) {
                continue;
            }

            double fillRatio =
                    polygonArea / Math.max(1.0, boxArea);

            if (fillRatio < 0.03 || fillRatio > 1.05) {
                continue;
            }

            Point center = polygonCenter(polygon);
            double dx = center.x - imageWidth / 2.0;
            double dy = center.y - imageHeight / 2.0;
            double distance = Math.sqrt(dx * dx + dy * dy);
            double maxDistance = Math.sqrt(
                    Math.pow(imageWidth / 2.0, 2)
                            + Math.pow(imageHeight / 2.0, 2)
            );

            double centerScore =
                    1.0 - Math.min(1.0, distance / maxDistance);

            double labelScore =
                    isMedicineLabel(label) ? 2.0 : 0.0;

            double fillScore =
                    1.0 - Math.min(
                            1.0,
                            Math.abs(fillRatio - 0.65)
                    );

            double score =
                    centerScore * 4.0
                            + labelScore
                            + fillScore * 2.0
                            + Math.min(
                            1.0,
                            polygonArea / (imageArea * 0.03)
                    );

            candidates.add(
                    new Candidate(
                            label,
                            normalizedBox,
                            pixelBox,
                            polygon,
                            score
                    )
            );
        }

        candidates.sort(
                (first, second) ->
                        Double.compare(
                                second.score,
                                first.score
                        )
        );

        if (candidates.size() > MAX_SEGMENTED_MEDICINES) {
            return new ArrayList<>(
                    candidates.subList(
                            0,
                            MAX_SEGMENTED_MEDICINES
                    )
            );
        }

        return candidates;
    }

    /**
     * A dokumentáció szerinti egyetlen helyes visszaszámítás.
     * A mask pontjai a bbox BELSEJÉBEN 0..1000-re normalizált [x,y] pontok.
     */
    private List<Point> convertBoxLocalMaskToPixels(
            int[] normalizedBox,
            JSONArray mask,
            int imageWidth,
            int imageHeight
    ) throws Exception {
        double ymin = normalizedBox[0];
        double xmin = normalizedBox[1];
        double ymax = normalizedBox[2];
        double xmax = normalizedBox[3];

        List<Point> result = new ArrayList<>();

        for (int i = 0; i < mask.length(); i++) {
            JSONArray pair = mask.optJSONArray(i);
            if (pair == null || pair.length() < 2) {
                continue;
            }

            double localX = clamp(
                    pair.optDouble(0, 0.0),
                    0.0,
                    1000.0
            );
            double localY = clamp(
                    pair.optDouble(1, 0.0),
                    0.0,
                    1000.0
            );

            double fullImageX1000 =
                    xmin + (localX / 1000.0) * (xmax - xmin);
            double fullImageY1000 =
                    ymin + (localY / 1000.0) * (ymax - ymin);

            double pixelX =
                    (fullImageX1000 / 1000.0) * (imageWidth - 1);
            double pixelY =
                    (fullImageY1000 / 1000.0) * (imageHeight - 1);

            Point point = new Point(
                    clamp(pixelX, 0.0, imageWidth - 1.0),
                    clamp(pixelY, 0.0, imageHeight - 1.0)
            );

            if (result.isEmpty()
                    || distance(result.get(result.size() - 1), point) >= 1.0) {
                result.add(point);
            }
        }

        if (result.size() >= 2
                && distance(result.get(0), result.get(result.size() - 1)) < 1.0) {
            result.remove(result.size() - 1);
        }

        return result;
    }

    private PixelBox normalizedBoxToPixelBox(
            int[] normalizedBox,
            int imageWidth,
            int imageHeight
    ) {
        int top = clampInt(
                Math.round(normalizedBox[0] / 1000f * (imageHeight - 1)),
                0,
                imageHeight - 1
        );
        int left = clampInt(
                Math.round(normalizedBox[1] / 1000f * (imageWidth - 1)),
                0,
                imageWidth - 1
        );
        int bottom = clampInt(
                Math.round(normalizedBox[2] / 1000f * (imageHeight - 1)),
                top + 1,
                imageHeight
        );
        int right = clampInt(
                Math.round(normalizedBox[3] / 1000f * (imageWidth - 1)),
                left + 1,
                imageWidth
        );

        return new PixelBox(left, top, right, bottom);
    }

    private String extractInteractionOutputText(
            String rawResponse
    ) throws Exception {
        JSONObject root = new JSONObject(rawResponse);

        String direct = root.optString("output_text", "");
        if (!direct.trim().isEmpty()) {
            return direct;
        }

        JSONArray steps = root.optJSONArray("steps");
        if (steps != null) {
            StringBuilder builder = new StringBuilder();

            for (int i = 0; i < steps.length(); i++) {
                JSONObject step = steps.optJSONObject(i);
                if (step == null
                        || !"model_output".equals(step.optString("type"))) {
                    continue;
                }

                JSONArray content = step.optJSONArray("content");
                if (content == null) {
                    continue;
                }

                for (int j = 0; j < content.length(); j++) {
                    JSONObject block = content.optJSONObject(j);
                    if (block != null
                            && "text".equals(block.optString("type"))) {
                        builder.append(block.optString("text", ""));
                    }
                }
            }

            if (builder.length() > 0) {
                return builder.toString();
            }
        }

        throw new Exception(
                "Az Interactions API válaszában nincs feldolgozható szöveg. Válasz: "
                        + shorten(rawResponse, 700)
        );
    }

    private JSONArray findEmbeddingValues(JSONObject object) {
        JSONObject embedding = object.optJSONObject("embedding");
        if (embedding != null) {
            JSONArray values = embedding.optJSONArray("values");
            if (values != null) {
                return values;
            }
        }

        JSONArray embeddings = object.optJSONArray("embeddings");
        if (embeddings != null && embeddings.length() > 0) {
            JSONObject first = embeddings.optJSONObject(0);
            if (first != null) {
                return first.optJSONArray("values");
            }
        }

        return null;
    }

    private EncodedImage encodeBitmap(
            Bitmap bitmap,
            Bitmap.CompressFormat format,
            int quality,
            int maxSide,
            String mimeType
    ) throws Exception {
        float scale = Math.min(
                1.0f,
                Math.min(
                        (float) maxSide / bitmap.getWidth(),
                        (float) maxSide / bitmap.getHeight()
                )
        );

        Bitmap output = bitmap;
        boolean scaled = false;

        if (scale < 1.0f) {
            int width = Math.max(
                    1,
                    Math.round(bitmap.getWidth() * scale)
            );
            int height = Math.max(
                    1,
                    Math.round(bitmap.getHeight() * scale)
            );

            output = Bitmap.createScaledBitmap(
                    bitmap,
                    width,
                    height,
                    true
            );
            scaled = output != bitmap;
        }

        try (ByteArrayOutputStream stream =
                     new ByteArrayOutputStream()) {
            boolean success = output.compress(
                    format,
                    quality,
                    stream
            );

            if (!success) {
                throw new Exception(
                        "A bitmap tömörítése sikertelen."
                );
            }

            byte[] bytes = stream.toByteArray();

            Log.d(
                    TAG,
                    "Kódolt kép="
                            + output.getWidth()
                            + "x"
                            + output.getHeight()
                            + "; byte="
                            + bytes.length
            );

            return new EncodedImage(
                    Base64.encodeToString(bytes, Base64.NO_WRAP),
                    mimeType
            );
        } finally {
            if (scaled && !output.isRecycled()) {
                output.recycle();
            }
        }
    }

    private String postJson(
            OkHttpClient client,
            String url,
            String json,
            String operation
    ) throws Exception {
        MediaType mediaType = MediaType.parse(
                "application/json; charset=utf-8"
        );

        RequestBody body = RequestBody.create(json, mediaType);

        Request request = new Request.Builder()
                .url(url)
                .addHeader("Content-Type", "application/json")
                .addHeader("x-goog-api-key", API_KEY)
                .addHeader("Api-Revision", "2026-05-20")
                .post(body)
                .build();

        long startedAt = SystemClock.elapsedRealtime();

        try (Response response = client.newCall(request).execute()) {
            String responseText = response.body() != null
                    ? response.body().string()
                    : "";

            Log.d(
                    TAG,
                    operation
                            + " HTTP-idő="
                            + formatSeconds(
                            millisecondsToSeconds(
                                    SystemClock.elapsedRealtime() - startedAt
                            )
                    )
                            + " s; HTTP="
                            + response.code()
            );

            if (!response.isSuccessful()) {
                throw new Exception(
                        "Gemini "
                                + operation
                                + " API hiba HTTP "
                                + response.code()
                                + ": "
                                + shorten(responseText, 1000)
                );
            }

            return responseText;
        }
    }

    private boolean isMedicineLabel(String label) {
        String value = label == null
                ? ""
                : label.toLowerCase(Locale.ROOT);

        return value.contains("medicine")
                || value.contains("pill")
                || value.contains("tablet")
                || value.contains("capsule")
                || value.contains("caplet")
                || value.contains("medication")
                || value.contains("drug")
                || value.contains("gyógyszer")
                || value.contains("tabletta")
                || value.contains("kapszula");
    }

    private double polygonArea(List<Point> points) {
        double area = 0.0;

        for (int i = 0; i < points.size(); i++) {
            Point current = points.get(i);
            Point next = points.get((i + 1) % points.size());
            area += current.x * next.y - next.x * current.y;
        }

        return Math.abs(area) * 0.5;
    }

    private Point polygonCenter(List<Point> points) {
        double sumX = 0.0;
        double sumY = 0.0;

        for (Point point : points) {
            sumX += point.x;
            sumY += point.y;
        }

        return new Point(
                sumX / points.size(),
                sumY / points.size()
        );
    }

    private double distance(Point first, Point second) {
        double dx = first.x - second.x;
        double dy = first.y - second.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private void validateApiKey() throws Exception {
        if (API_KEY == null
                || API_KEY.trim().isEmpty()
                || API_KEY.startsWith("IDE_")) {
            throw new Exception(
                    "Nincs beállítva az új Gemini API-kulcs."
            );
        }
    }

    private void validateBitmap(Bitmap bitmap) throws Exception {
        if (bitmap == null) {
            throw new Exception("A bitmap null.");
        }
        if (bitmap.isRecycled()) {
            throw new Exception("A bitmap már fel lett szabadítva.");
        }
        if (bitmap.getWidth() <= 0 || bitmap.getHeight() <= 0) {
            throw new Exception("Hibás bitmap méret.");
        }
    }

    private String cleanJsonText(String text) {
        String value = text == null ? "" : text.trim();

        if (value.startsWith("```")) {
            value = value.replaceFirst("^```(?:json)?\\s*", "");
            value = value.replaceFirst("\\s*```$", "");
        }

        return value.trim();
    }

    private String shorten(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        if (text.length() <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + "...";
    }

    private int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private double millisecondsToSeconds(long milliseconds) {
        return milliseconds / 1000.0;
    }

    private String formatSeconds(double seconds) {
        return String.format(
                Locale.getDefault(),
                "%.3f",
                seconds
        );
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public static String getUserFriendlyErrorMessage(
            Throwable error
    ) {
        Throwable current =
                error;

        int causeDepth =
                0;

        while (current != null
                && causeDepth < 6) {
            String message =
                    current.getMessage() == null
                            ? ""
                            : current.getMessage();

            String normalized =
                    message.toLowerCase(
                            Locale.ROOT
                    );

            if (current instanceof java.io.InterruptedIOException
                    || normalized.contains("timeout")
                    || normalized.contains("timed out")
                    || normalized.contains("időtúllépés")) {
                return "A képelemzés túl sokáig tartott. "
                        + "Ellenőrizd az internetkapcsolatot, "
                        + "majd próbáld újra.";
            }

            if (current instanceof java.io.IOException
                    || normalized.contains("unable to resolve host")
                    || normalized.contains("failed to connect")
                    || normalized.contains("connection refused")
                    || normalized.contains("network is unreachable")) {
                return "Nem sikerült kapcsolódni az online "
                        + "felismerő szolgáltatáshoz. Ellenőrizd "
                        + "az internetkapcsolatot, majd próbáld újra.";
            }

            if (normalized.contains("http 429")
                    || normalized.contains("quota")
                    || normalized.contains("rate limit")
                    || normalized.contains("resource_exhausted")) {
                return "A felismerő szolgáltatás átmenetileg "
                        + "túlterhelt. Várj egy percet, majd "
                        + "próbáld újra.";
            }

            if (normalized.contains("http 500")
                    || normalized.contains("http 502")
                    || normalized.contains("http 503")
                    || normalized.contains("http 504")
                    || normalized.contains("internal server")) {
                return "Az online felismerő szolgáltatás "
                        + "átmenetileg nem elérhető. Próbáld "
                        + "újra néhány perc múlva.";
            }

            if (normalized.contains("http 401")
                    || normalized.contains("http 403")
                    || normalized.contains("api-kulcs")
                    || normalized.contains("api key")
                    || normalized.contains("permission_denied")
                    || normalized.contains("unauthorized")) {
                return "Az online felismerő szolgáltatás "
                        + "jelenleg nem használható. Kérlek, "
                        + "próbáld újra később.";
            }

            if (normalized.contains("http 400")
                    || normalized.contains("invalid_request")) {
                return "Az online képelemző szolgáltatás "
                        + "nem tudta feldolgozni a képet. "
                        + "Próbáld újra egy másik fényképpel.";
            }

            if (normalized.contains("boxes tömb")
                    || normalized.contains("gyógyszerszegmens")
                    || normalized.contains("szegment")
                    || normalized.contains("mask")) {
                return "Nem sikerült megfelelően elkülöníteni a "
                        + "gyógyszert a háttértől. Tedd a gyógyszert "
                        + "a lap közepére, kerüld az erős árnyékot, "
                        + "majd készíts új képet.";
            }

            if (normalized.contains("embedding")
                    || normalized.contains("vector")
                    || normalized.contains("dimenzió")
                    || normalized.contains("dimension")) {
                return "A gyógyszer összehasonlításához szükséges "
                        + "képi jellemzők nem készültek el. "
                        + "Próbáld újra egy élesebb képpel.";
            }

            if (normalized.contains("interactions api")
                    || normalized.contains("gemini")
                    || normalized.contains("feldolgozható szöveg")
                    || normalized.contains("hibás választ")
                    || normalized.contains("válaszban")) {
                return "A gyógyszer online elemzése most nem "
                        + "sikerült. Ellenőrizd, hogy a gyógyszer "
                        + "jól látható legyen, majd próbáld újra.";
            }

            current =
                    current.getCause();

            causeDepth++;
        }

        return null;
    }

    private static final class EncodedImage {
        final String base64;
        final String mimeType;

        EncodedImage(String base64, String mimeType) {
            this.base64 = base64;
            this.mimeType = mimeType;
        }
    }

    private static final class Candidate {
        final String label;
        final int[] normalizedBox;
        final PixelBox pixelBox;
        final List<Point> polygonPixels;
        final double score;

        Candidate(
                String label,
                int[] normalizedBox,
                PixelBox pixelBox,
                List<Point> polygonPixels,
                double score
        ) {
            this.label = label;
            this.normalizedBox = normalizedBox;
            this.pixelBox = pixelBox;
            this.polygonPixels = polygonPixels;
            this.score = score;
        }
    }

    public static final class PixelBox {
        public final int left;
        public final int top;
        public final int right;
        public final int bottom;

        public PixelBox(
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
            return Math.max(0, right - left);
        }

        public int height() {
            return Math.max(0, bottom - top);
        }

        public double area() {
            return (double) width() * height();
        }

        @Override
        public String toString() {
            return "["
                    + left + ","
                    + top + ","
                    + right + ","
                    + bottom + "]";
        }
    }

    public static final class SegmentationResult {
        private final String label;
        private final int[] normalizedBox;
        private final PixelBox pixelBox;
        private final List<Point> polygonPixels;
        private final String structuredJson;
        private final String rawApiResponse;
        private final double runtimeSeconds;
        private final String model;

        SegmentationResult(
                String label,
                int[] normalizedBox,
                PixelBox pixelBox,
                List<Point> polygonPixels,
                String structuredJson,
                String rawApiResponse,
                double runtimeSeconds,
                String model
        ) {
            this.label = label;
            this.normalizedBox = normalizedBox.clone();
            this.pixelBox = pixelBox;
            this.polygonPixels = new ArrayList<>(polygonPixels);
            this.structuredJson = structuredJson;
            this.rawApiResponse = rawApiResponse;
            this.runtimeSeconds = runtimeSeconds;
            this.model = model;
        }

        public String getLabel() {
            return label;
        }

        public int[] getNormalizedBox() {
            return normalizedBox.clone();
        }

        public PixelBox getPixelBox() {
            return pixelBox;
        }

        public List<Point> getPolygonPixels() {
            return new ArrayList<>(polygonPixels);
        }

        public String getStructuredJson() {
            return structuredJson;
        }

        public String getRawApiResponse() {
            return rawApiResponse;
        }

        public double getRuntimeSeconds() {
            return runtimeSeconds;
        }

        public String getModel() {
            return model;
        }
    }
}