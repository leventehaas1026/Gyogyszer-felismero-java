package com.example.kmd5a9_szakmaigyakorlat;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.exifinterface.media.ExifInterface;
import androidx.fragment.app.Fragment;

import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.imageview.ShapeableImageView;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.UserInfo;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.SetOptions;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

public class ProfileFragment extends Fragment {

    private ShapeableImageView profileBigAvatar;
    private ActivityResultLauncher<Intent> imagePickerLauncher;

    public ProfileFragment() {}

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == android.app.Activity.RESULT_OK && result.getData() != null) {
                        saveProfileImageToBase64(result.getData().getData());
                    }
                });
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_profile, container, false);

        profileBigAvatar = view.findViewById(R.id.profile_big_avatar);
        Button btnChangeAvatar = view.findViewById(R.id.btn_change_avatar);
        Button btnChangePassword = view.findViewById(R.id.btn_change_password);

        loadProfileImageFromFirestore();

        btnChangeAvatar.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            imagePickerLauncher.launch(intent);
        });

        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            boolean isGoogleUser = false;
            for (UserInfo profile : user.getProviderData()) {
                if (profile.getProviderId().equals("google.com")) {
                    isGoogleUser = true;
                    break;
                }
            }

            if (!isGoogleUser) {
                btnChangePassword.setVisibility(View.VISIBLE);
                btnChangePassword.setOnClickListener(v -> showPasswordChangeDialog());
            }
        }

        return view;
    }

    private void saveProfileImageToBase64(Uri imageUri) {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user == null) return;

        try {
            // 1. Kép betöltése és elforgatás korrekciója
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContext().getContentResolver(), imageUri);

            InputStream input = getContext().getContentResolver().openInputStream(imageUri);
            ExifInterface exif = new ExifInterface(input);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            Matrix matrix = new Matrix();
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90: matrix.postRotate(90); break;
                case ExifInterface.ORIENTATION_ROTATE_180: matrix.postRotate(180); break;
                case ExifInterface.ORIENTATION_ROTATE_270: matrix.postRotate(270); break;
            }
            Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

            int originalWidth = rotatedBitmap.getWidth();
            int originalHeight = rotatedBitmap.getHeight();
            int targetSize = 300;

            int newWidth, newHeight;
            if (originalWidth > originalHeight) {
                newWidth = targetSize;
                newHeight = (int) (targetSize * ((float) originalHeight / originalWidth));
            } else {
                newHeight = targetSize;
                newWidth = (int) (targetSize * ((float) originalWidth / originalHeight));
            }

            // Képarányok megtartása
            Bitmap resized = Bitmap.createScaledBitmap(rotatedBitmap, newWidth, newHeight, true);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            resized.compress(Bitmap.CompressFormat.JPEG, 70, baos);
            byte[] imageBytes = baos.toByteArray();
            String base64Image = Base64.encodeToString(imageBytes, Base64.DEFAULT);

            // kép mentése
            Map<String, Object> userData = new HashMap<>();
            userData.put("profileImageBase64", base64Image);

            FirebaseFirestore.getInstance().collection("users").document(user.getUid())
                    .set(userData, SetOptions.merge())
                    .addOnSuccessListener(aVoid -> {
                        profileBigAvatar.setImageBitmap(resized);
                        Toast.makeText(getContext(), "Profilkép mentve!", Toast.LENGTH_SHORT).show();
                    })
                    .addOnFailureListener(e -> Toast.makeText(getContext(), "Hiba: " + e.getMessage(), Toast.LENGTH_SHORT).show());

        } catch (Exception e) {
            Toast.makeText(getContext(), "Képfeldolgozási hiba: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void loadProfileImageFromFirestore() {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user == null) return;

        FirebaseFirestore.getInstance().collection("users").document(user.getUid())
                .get()
                .addOnSuccessListener(documentSnapshot -> {
                    String base64 = documentSnapshot.getString("profileImageBase64");
                    if (base64 != null && !base64.isEmpty()) {
                        byte[] decodedBytes = Base64.decode(base64, Base64.DEFAULT);
                        Bitmap bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
                        profileBigAvatar.setImageBitmap(bitmap);
                    }
                });
    }

    private void showPasswordChangeDialog() {
        EditText newPasswordInput = new EditText(getContext());
        newPasswordInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        newPasswordInput.setHint("Új jelszó (min. 6 karakter)");

        new MaterialAlertDialogBuilder(requireContext())
                .setTitle("Jelszó módosítása")
                .setView(newPasswordInput)
                .setPositiveButton("Módosítás", (dialog, which) -> {
                    String newPassword = newPasswordInput.getText().toString();
                    if (newPassword.length() >= 6) {
                        FirebaseAuth.getInstance().getCurrentUser().updatePassword(newPassword)
                                .addOnCompleteListener(task -> {
                                    if (task.isSuccessful()) {
                                        Toast.makeText(getContext(), "Jelszó frissítve!", Toast.LENGTH_SHORT).show();
                                    } else {
                                        Toast.makeText(getContext(), "Hiba: " + task.getException().getMessage(), Toast.LENGTH_LONG).show();
                                    }
                                });
                    } else {
                        Toast.makeText(getContext(), "A jelszó legalább 6 karakter legyen!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Mégse", null)
                .show();
    }
}