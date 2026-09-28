package com.example.kmd5a9_szakmaigyakorlat;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import com.google.android.material.card.MaterialCardView;

public class HomeFragment extends Fragment {

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_home, container, false);

        MaterialCardView receptCard = view.findViewById(R.id.recept_card_btn);
        receptCard.setOnClickListener(v -> {
            if (getActivity() instanceof LandingPage) {
                ((LandingPage) getActivity()).changeTab(R.id.nav_scan);
            }
        });

        MaterialCardView gyogyszerCard = view.findViewById(R.id.gyogyszer_card_btn);

        gyogyszerCard.setOnClickListener(v -> {
            if (getActivity() instanceof LandingPage) {
                ((LandingPage) getActivity()).changeTab(R.id.nav_recept);
            }
        });

        return view;
    }
}