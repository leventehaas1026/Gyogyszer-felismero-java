package com.example.kmd5a9_szakmaigyakorlat;

import android.os.Bundle;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import com.bumptech.glide.Glide;
import com.github.chrisbanes.photoview.PhotoView;

public class ImageDetailActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_image_detail);

        PhotoView photoView = findViewById(R.id.photo_view);
        String imageUri = getIntent().getStringExtra("image_uri");

        Glide.with(this).load(imageUri).into(photoView);
    }
}