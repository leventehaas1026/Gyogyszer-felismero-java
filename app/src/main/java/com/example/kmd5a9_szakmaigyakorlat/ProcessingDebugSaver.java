package com.example.kmd5a9_szakmaigyakorlat;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.RequiresApi;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;

/**
 * A feldolgozás köztes képeit és debugadatait menti.
 *
 * Végleges mappastruktúra Android 10+ rendszeren:
 *
 * Download/MedScan/
 * ├── Images/
 * │   ├── Pills/
 * │   └── Prescriptions/
 * └── Debug/
 *     ├── Images/<sessionId>/
 *     └── Data/<sessionId>/
 *
 * Egy ProcessingDebugSaver példány egyetlen feldolgozási munkamenetet jelent.
 * Emiatt az ugyanazon példánnyal mentett összes kép, JSON és TXT ugyanahhoz
 * az időbélyeges sessionId-hez kerül.
 */
public final class ProcessingDebugSaver {

    private static final String TAG = "ProcessingDebugSaver";


    /*
     * A JSON/TXT fájlokat a MediaStore csak a Download vagy Documents
     * elsődleges könyvtár alatt engedi. Ezért a teljes Debug struktúra
     * a Download/MedScan/Debug alá kerül. A felhasználói képek is ugyanennek a Download/MedScan gyökérnek az Images almappáiba kerülnek.
     */
    private static final String MEDSCAN_DEBUG_ROOT =
            Environment.DIRECTORY_DOWNLOADS + "/MedScan";

    private final Context context;
    private final String sessionId;
    private final String imageRelativePath;
    private final String dataRelativePath;

    public ProcessingDebugSaver(Context context) {
        if (context == null) {
            throw new IllegalArgumentException(
                    "A Context nem lehet null."
            );
        }

        this.context = context.getApplicationContext();

        String timestamp =
                new SimpleDateFormat(
                        "yyyyMMdd_HHmmss_SSS",
                        Locale.ROOT
                ).format(new Date());

        String suffix =
                UUID.randomUUID()
                        .toString()
                        .substring(0, 8);

        sessionId = timestamp + "_" + suffix;

        imageRelativePath =
                MEDSCAN_DEBUG_ROOT
                        + "/Debug/Images/"
                        + sessionId;

        dataRelativePath =
                MEDSCAN_DEBUG_ROOT
                        + "/Debug/Data/"
                        + sessionId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getImageRelativePath() {
        return imageRelativePath;
    }

    public String getDataRelativePath() {
        return dataRelativePath;
    }

    public String saveBitmapPng(
            String filename,
            Bitmap bitmap
    ) throws Exception {
        validateBitmap(filename, bitmap);

        String safeName =
                ensureExtension(
                        sanitizeFilename(filename),
                        ".png"
                );

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return saveBitmapWithMediaStore(
                    safeName,
                    bitmap,
                    Bitmap.CompressFormat.PNG,
                    100,
                    "image/png"
            );
        }

        return saveBitmapLegacy(
                safeName,
                bitmap,
                Bitmap.CompressFormat.PNG,
                100
        );
    }

    /**
     * JPEG-mentés a normál debugképekhez.
     */
    public String saveBitmapJpeg(
            String filename,
            Bitmap bitmap,
            int quality
    ) throws Exception {
        validateBitmap(filename, bitmap);

        int safeQuality = Math.max(1, Math.min(100, quality));

        String safeName =
                ensureJpegExtension(
                        sanitizeFilename(filename)
                );

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return saveBitmapWithMediaStore(
                    safeName,
                    bitmap,
                    Bitmap.CompressFormat.JPEG,
                    safeQuality,
                    "image/jpeg"
            );
        }

        return saveBitmapLegacy(
                safeName,
                bitmap,
                Bitmap.CompressFormat.JPEG,
                safeQuality
        );
    }

    /**
     * JSON, TXT és minden egyéb szöveges debugadat mentése.
     * Android 10+ alatt ugyanazon Pictures/MedScan gyökérben marad.
     */
    public String saveText(
            String filename,
            String content,
            String mimeType
    ) throws Exception {
        String safeName = sanitizeFilename(filename);

        byte[] bytes =
                (content == null ? "" : content)
                        .getBytes(StandardCharsets.UTF_8);

        String resolvedMimeType =
                resolveTextMimeType(
                        safeName,
                        mimeType
                );

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return saveDataWithMediaStore(
                    safeName,
                    bytes,
                    resolvedMimeType
            );
        }

        File directory = getLegacyDirectory(false);
        File target = new File(directory, safeName);

        try (OutputStream outputStream =
                     new FileOutputStream(target)) {
            outputStream.write(bytes);
            outputStream.flush();
        }

        Log.d(
                TAG,
                "Debugadat mentve: "
                        + target.getAbsolutePath()
        );

        return target.getAbsolutePath();
    }

    private void validateBitmap(
            String filename,
            Bitmap bitmap
    ) throws Exception {
        if (bitmap == null
                || bitmap.isRecycled()
                || bitmap.getWidth() <= 0
                || bitmap.getHeight() <= 0) {
            throw new Exception(
                    "Nem menthető hibás bitmap: "
                            + filename
            );
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private String saveBitmapWithMediaStore(
            String filename,
            Bitmap bitmap,
            Bitmap.CompressFormat format,
            int quality,
            String mimeType
    ) throws Exception {
        ContentValues values = new ContentValues();

        values.put(
                MediaStore.Images.Media.DISPLAY_NAME,
                filename
        );
        values.put(
                MediaStore.Images.Media.MIME_TYPE,
                mimeType
        );
        values.put(
                MediaStore.Images.Media.RELATIVE_PATH,
                imageRelativePath
        );
        values.put(
                MediaStore.Images.Media.IS_PENDING,
                1
        );

        Uri uri =
                context.getContentResolver().insert(
                        MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                        values
                );

        if (uri == null) {
            throw new Exception(
                    "Nem sikerült debugkép URI-t létrehozni: "
                            + filename
            );
        }

        boolean completed = false;

        try (OutputStream outputStream =
                     context.getContentResolver()
                             .openOutputStream(uri)) {
            if (outputStream == null) {
                throw new Exception(
                        "Nem nyitható meg a debugkép: "
                                + filename
                );
            }

            boolean success =
                    bitmap.compress(
                            format,
                            quality,
                            outputStream
                    );

            if (!success) {
                throw new Exception(
                        "A debugkép tömörítése sikertelen: "
                                + filename
                );
            }

            outputStream.flush();
            completed = true;

        } finally {
            finishOrDeleteMediaStoreItem(
                    uri,
                    completed
            );
        }

        Log.d(
                TAG,
                "Debugkép mentve: "
                        + imageRelativePath
                        + "/"
                        + filename
        );

        return uri.toString();
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private String saveDataWithMediaStore(
            String filename,
            byte[] bytes,
            String mimeType
    ) throws Exception {
        ContentValues values = new ContentValues();

        values.put(
                MediaStore.MediaColumns.DISPLAY_NAME,
                filename
        );
        values.put(
                MediaStore.MediaColumns.MIME_TYPE,
                mimeType
        );
        values.put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                dataRelativePath
        );
        values.put(
                MediaStore.MediaColumns.IS_PENDING,
                1
        );

        Uri collection =
                MediaStore.Files.getContentUri(
                        MediaStore.VOLUME_EXTERNAL_PRIMARY
                );

        Uri uri =
                context.getContentResolver().insert(
                        collection,
                        values
                );

        if (uri == null) {
            throw new Exception(
                    "Nem sikerült debugadat URI-t létrehozni: "
                            + filename
            );
        }

        boolean completed = false;

        try (OutputStream outputStream =
                     context.getContentResolver()
                             .openOutputStream(uri)) {
            if (outputStream == null) {
                throw new Exception(
                        "Nem nyitható meg a debugadat: "
                                + filename
                );
            }

            outputStream.write(bytes);
            outputStream.flush();
            completed = true;

        } finally {
            finishOrDeleteMediaStoreItem(
                    uri,
                    completed
            );
        }

        Log.d(
                TAG,
                "Debugadat mentve: "
                        + dataRelativePath
                        + "/"
                        + filename
        );

        return uri.toString();
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private void finishOrDeleteMediaStoreItem(
            Uri uri,
            boolean completed
    ) {
        if (completed) {
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
        } else {
            context.getContentResolver()
                    .delete(
                            uri,
                            null,
                            null
                    );
        }
    }

    private String saveBitmapLegacy(
            String filename,
            Bitmap bitmap,
            Bitmap.CompressFormat format,
            int quality
    ) throws Exception {
        File directory = getLegacyDirectory(true);
        File target = new File(directory, filename);

        try (OutputStream outputStream =
                     new FileOutputStream(target)) {
            boolean success =
                    bitmap.compress(
                            format,
                            quality,
                            outputStream
                    );

            if (!success) {
                throw new Exception(
                        "A debugkép mentése sikertelen: "
                                + filename
                );
            }

            outputStream.flush();
        }

        Log.d(
                TAG,
                "Debugkép mentve: "
                        + target.getAbsolutePath()
        );

        return target.getAbsolutePath();
    }

    /**
     * Android 7–9 alatt az alkalmazás saját külső Pictures mappája használható
     * külön tárhelyengedély nélkül.
     */
    private File getLegacyDirectory(
            boolean imageDirectory
    ) throws Exception {
        File base =
                context.getExternalFilesDir(
                        Environment.DIRECTORY_DOWNLOADS
                );

        if (base == null) {
            throw new Exception(
                    "A külső Pictures alkalmazásmappa nem érhető el."
            );
        }

        String relative =
                imageDirectory
                        ? "MedScan/Debug/Images/" + sessionId
                        : "MedScan/Debug/Data/" + sessionId;

        File directory =
                new File(
                        base,
                        relative
                );

        if (!directory.exists()
                && !directory.mkdirs()) {
            throw new Exception(
                    "Nem hozható létre a debugmappa: "
                            + directory
            );
        }

        return directory;
    }

    private String resolveTextMimeType(
            String filename,
            String mimeType
    ) {
        if (mimeType != null
                && !mimeType.trim().isEmpty()) {
            return mimeType.trim();
        }

        String lower =
                filename.toLowerCase(Locale.ROOT);

        if (lower.endsWith(".json")) {
            return "application/json";
        }

        return "text/plain";
    }

    private String sanitizeFilename(
            String filename
    ) {
        String value =
                filename == null
                        ? "artifact"
                        : filename.trim();

        if (value.isEmpty()) {
            value = "artifact";
        }

        return value.replaceAll(
                "[^a-zA-Z0-9._-]",
                "_"
        );
    }

    private String ensureExtension(
            String filename,
            String extension
    ) {
        if (filename.toLowerCase(Locale.ROOT)
                .endsWith(extension)) {
            return filename;
        }

        return filename + extension;
    }

    private String ensureJpegExtension(
            String filename
    ) {
        String lower =
                filename.toLowerCase(Locale.ROOT);

        if (lower.endsWith(".jpg")
                || lower.endsWith(".jpeg")) {
            return filename;
        }

        return filename + ".jpg";
    }
}