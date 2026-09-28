package com.example.kmd5a9_szakmaigyakorlat;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class DatabaseManager {
    private SQLiteDatabase db;
    private static final String DB_NAME = "med_database.db";

    public DatabaseManager(Context context) {
        try {
            File dbFile = context.getDatabasePath(DB_NAME);
            // Ha még nem létezik a belső mappában, átmásoljuk az assets-ből
            if (!dbFile.exists()) {
                InputStream is = context.getAssets().open(DB_NAME);
                FileOutputStream fos = new FileOutputStream(dbFile);
                byte[] buffer = new byte[1024];
                int length;
                while ((length = is.read(buffer)) > 0) fos.write(buffer, 0, length);
                fos.close();
                is.close();
                Log.d("DatabaseManager", "Adatbázis másolva az assets-ből.");
            }
            db = SQLiteDatabase.openDatabase(dbFile.getPath(), null, SQLiteDatabase.OPEN_READONLY);
        } catch (Exception e) {
            Log.e("DatabaseManager", "Hiba az adatbázis megnyitásakor: " + e.getMessage());
        }
    }

    public Cursor getAllMedications() {
        return db.rawQuery("SELECT med_name, left_vector, right_vector FROM medications", null);
    }

    // Segédmetódus: String (vesszővel elválasztott) -> float[]
    public static float[] stringToFloatArray(String s) {
        String[] parts = s.split(",");
        float[] result = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Float.parseFloat(parts[i]);
        }
        return result;
    }

    // Segédmetódus: két float tömb összefűzése (hogy illeszkedjen a Python logikádhoz)
    public static float[] combineVectors(String left, String right) {
        float[] v1 = stringToFloatArray(left);
        float[] v2 = stringToFloatArray(right);
        float[] combined = new float[v1.length + v2.length];
        System.arraycopy(v1, 0, combined, 0, v1.length);
        System.arraycopy(v2, 0, combined, v1.length, v2.length);
        return combined;
    }

    public void close() {
        if (db != null) db.close();
    }
}