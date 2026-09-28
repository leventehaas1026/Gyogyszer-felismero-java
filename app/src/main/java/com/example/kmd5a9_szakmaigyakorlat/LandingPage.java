package com.example.kmd5a9_szakmaigyakorlat;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.ImageView;
import android.widget.RadioGroup;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.imageview.ShapeableImageView;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;
import com.example.kmd5a9_szakmaigyakorlat.BuildConfig;

public class LandingPage extends AppCompatActivity {

    private Fragment homeFragment = new HomeFragment();
    private Fragment scanFragment = new ScanFragment();
    private Fragment pillScanFragment = new PillScanFragment();
    private Fragment historyFragment = new HistoryFragment();
    private Fragment profileFragment = new ProfileFragment();
    private Fragment activeFragment = homeFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_landing_page);

        setupProfileAvatar();

        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);

        // Fragmentek inicializálása
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container, profileFragment, "4").hide(profileFragment).commit();
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container, historyFragment, "3").hide(historyFragment).commit();
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container, scanFragment, "2").hide(scanFragment).commit();
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container, homeFragment, "1").commit();
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container, pillScanFragment, "5").hide(pillScanFragment).commit();

        bottomNavigationView.setOnItemSelectedListener(item -> {
            int id = item.getItemId();

            if (id == R.id.nav_home) {
                switchFragment(homeFragment);
                return true;
            } else if (id == R.id.nav_scan) {
                switchFragment(scanFragment);
                return true;
            } else if (id == R.id.nav_recept) {
                switchFragment(pillScanFragment);
                return true;
            } else if (id == R.id.nav_history) {
                switchFragment(historyFragment);
                return true;
            }
            return false;
        });

        ImageView settingsBtn = findViewById(R.id.toolbar_settings);
        if (settingsBtn != null) {
            settingsBtn.setOnClickListener(v -> openSettings());
        }
    }

    private void switchFragment(Fragment target) {
        getSupportFragmentManager().beginTransaction().hide(activeFragment).show(target).commit();
        activeFragment = target;
    }

    private void setupProfileAvatar() {
        ShapeableImageView profileAvatar = findViewById(R.id.toolbar_profile_avatar);
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();

        if (user != null) {
            // Valós idejű figyelő a profilkép változására
            FirebaseFirestore.getInstance().collection("users").document(user.getUid())
                    .addSnapshotListener((snapshot, e) -> {
                        if (snapshot != null && snapshot.exists()) {
                            String base64 = snapshot.getString("profileImageBase64");
                            if (base64 != null && !base64.isEmpty()) {
                                byte[] decodedBytes = Base64.decode(base64, Base64.DEFAULT);
                                Bitmap bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
                                profileAvatar.setImageBitmap(bitmap);
                            }
                        }
                    });
        }

        profileAvatar.setOnClickListener(v -> {
            if (activeFragment != profileFragment) {
                switchFragment(profileFragment);
            }
        });
    }

    private void openSettings() {
        View dialogView = getLayoutInflater().inflate(R.layout.dialog_settings, null);

        android.widget.TextView tvVersion = dialogView.findViewById(R.id.tvVersion);
        if (tvVersion != null) {
            String ver = "Verzió: " + BuildConfig.VERSION_NAME;
            tvVersion.setText(ver);
        }

        com.google.android.material.dialog.MaterialAlertDialogBuilder builder =
                new com.google.android.material.dialog.MaterialAlertDialogBuilder(this);
        builder.setTitle("Beállítások").setView(dialogView);
        androidx.appcompat.app.AlertDialog dialog = builder.create();

        RadioGroup radioGroup = dialogView.findViewById(R.id.themeRadioGroup);
        int mode = androidx.appcompat.app.AppCompatDelegate.getDefaultNightMode();
        if (mode == androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO) radioGroup.check(R.id.rbLight);
        else if (mode == androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES) radioGroup.check(R.id.rbDark);
        else radioGroup.check(R.id.rbSystem);

        radioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rbLight) androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO);
            else if (checkedId == R.id.rbDark) androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES);
            else androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
            dialog.dismiss();
        });

        dialogView.findViewById(R.id.btnSignOut).setOnClickListener(v -> {
            dialog.dismiss();
            performSignOut();
        });
        dialog.show();
    }

    private void performSignOut() {
        FirebaseAuth.getInstance().signOut();
        com.google.android.gms.auth.api.signin.GoogleSignIn.getClient(this,
                new com.google.android.gms.auth.api.signin.GoogleSignInOptions.Builder(
                        com.google.android.gms.auth.api.signin.GoogleSignInOptions.DEFAULT_SIGN_IN).build()).signOut();
        Intent intent = new Intent(LandingPage.this, AuthActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }

    public void changeTab(int menuItemId) {
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);
        if (bottomNavigationView != null) bottomNavigationView.setSelectedItemId(menuItemId);
    }
}