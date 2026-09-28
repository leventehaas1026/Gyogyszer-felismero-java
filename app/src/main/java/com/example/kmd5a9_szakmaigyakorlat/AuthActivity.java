package com.example.kmd5a9_szakmaigyakorlat;

import static java.security.AccessController.getContext;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.api.ApiException;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.AuthCredential;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.GoogleAuthProvider;

public class AuthActivity extends AppCompatActivity {

    private TextInputEditText etEmail, etPassword;
    private MaterialButton btnMainAction, btnToggleMode, btnGoogleSignIn;
    private TextView tvTitle;

    private FirebaseAuth mAuth;
    private GoogleSignInClient mGoogleSignInClient;
    private final int RC_SIGN_IN = 9001;
    private boolean isLoginMode = true;
    private TextView tvForgotPassword;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_auth);

        mAuth = FirebaseAuth.getInstance();

        // Google SignIn konfiguráció
        GoogleSignInOptions gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestIdToken(getString(R.string.default_web_client_id))
                .requestEmail()
                .build();
        mGoogleSignInClient = GoogleSignIn.getClient(this, gso);

        etEmail = findViewById(R.id.etEmail);
        etPassword = findViewById(R.id.etPassword);
        btnMainAction = findViewById(R.id.btnMainAction);
        btnToggleMode = findViewById(R.id.btnToggleMode);
        btnGoogleSignIn = findViewById(R.id.btnGoogleSignIn); // Feltételezve, hogy hozzáadtad XML-be
        tvTitle = findViewById(R.id.tvTitle);

        btnToggleMode.setOnClickListener(v -> {
            isLoginMode = !isLoginMode;
            if (isLoginMode) {
                tvTitle.setText("Bejelentkezés");
                btnMainAction.setText("Bejelentkezés");
                btnToggleMode.setText("Nincs még fiókom? Regisztráció");
            } else {
                tvTitle.setText("Regisztráció");
                btnMainAction.setText("Fiók létrehozása");
                btnToggleMode.setText("Már van fiókom? Bejelentkezés");
            }
        });

        btnMainAction.setOnClickListener(v -> {
            String email = etEmail.getText().toString().trim();
            String password = etPassword.getText().toString().trim();
            if (email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "Kérlek, tölts ki minden mezőt!", Toast.LENGTH_SHORT).show();
                return;
            }
            if (isLoginMode) loginUser(email, password);
            else registerUser(email, password);
        });

        btnGoogleSignIn.setOnClickListener(v -> {
            Intent signInIntent = mGoogleSignInClient.getSignInIntent();
            startActivityForResult(signInIntent, RC_SIGN_IN);
        });

        tvForgotPassword = findViewById(R.id.tvForgotPassword);

        // Kattintás esemény az elfelejtett jelszóhoz
        tvForgotPassword.setOnClickListener(v -> {
            String email = etEmail.getText().toString().trim();

            if (email.isEmpty()) {
                etEmail.setError("Kérlek, add meg az e-mail címedet!");
                Toast.makeText(this, "Add meg az e-mail címedet a visszaállításhoz", Toast.LENGTH_SHORT).show();
            } else {
                sendPasswordResetEmail(email);
            }
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == RC_SIGN_IN) {
            try {
                GoogleSignInAccount account = GoogleSignIn.getSignedInAccountFromIntent(data).getResult(ApiException.class);
                firebaseAuthWithGoogle(account.getIdToken());
            } catch (ApiException e) {
                Toast.makeText(this, "Google bejelentkezés sikertelen", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void sendPasswordResetEmail(String email) {
        FirebaseAuth auth = FirebaseAuth.getInstance();

        auth.sendPasswordResetEmail(email)
                .addOnCompleteListener(task -> {
                    if (task.isSuccessful()) {
                        // Sikeres küldés: az e-mail létezik és kiküldtük az utasítást
                        Toast.makeText(this, "Ellenőrizd az e-mail fiókodat!", Toast.LENGTH_LONG).show();
                    } else {
                        // Ha hiba történt, mert nem létezik az email
                        Exception e = task.getException();
                        if (e instanceof com.google.firebase.auth.FirebaseAuthInvalidUserException) {
                            etEmail.setError("Ez az e-mail cím nincs regisztrálva!");
                            Toast.makeText(this, "Nincs ilyen e-mail címmel regisztrált felhasználó.", Toast.LENGTH_LONG).show();
                        } else {
                            // Egyéb hiba (pl. hálózati probléma)
                            Toast.makeText(this, "Hiba: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        }
                    }
                });
    }
    private void firebaseAuthWithGoogle(String idToken) {
        AuthCredential credential = GoogleAuthProvider.getCredential(idToken, null);
        mAuth.signInWithCredential(credential)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful()) goToLandingPage();
                    else Toast.makeText(this, "Hitelesítési hiba", Toast.LENGTH_SHORT).show();
                });
    }

    private void loginUser(String email, String password) {
        mAuth.signInWithEmailAndPassword(email, password)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful()) goToLandingPage();
                    else Toast.makeText(this, "Hiba: " + task.getException().getMessage(), Toast.LENGTH_LONG).show();
                });
    }

    private void registerUser(String email, String password) {
        mAuth.createUserWithEmailAndPassword(email, password)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful()) goToLandingPage();
                    else Toast.makeText(this, "Hiba: " + task.getException().getMessage(), Toast.LENGTH_LONG).show();
                });
    }

    private void goToLandingPage() {
        startActivity(new Intent(AuthActivity.this, LandingPage.class));
        finish();
    }
}