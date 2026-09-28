package com.example.kmd5a9_szakmaigyakorlat;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.checkbox.MaterialCheckBox;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class ReceptAdapter
        extends RecyclerView.Adapter<ReceptAdapter.ReceptViewHolder> {

    public interface SelectionListener {
        void onSelectionChanged(
                int selectedCount,
                boolean selectionMode
        );
    }

    private final Context context;
    private final List<Recept> receptList;
    private final SelectionListener selectionListener;
    private final Set<String> selectedDocumentIds =
            new HashSet<>();
    private boolean selectionMode;

    public ReceptAdapter(
            Context context,
            List<Recept> receptList,
            SelectionListener selectionListener
    ) {
        this.context = context;
        this.receptList = receptList;
        this.selectionListener = selectionListener;
    }

    @NonNull
    @Override
    public ReceptViewHolder onCreateViewHolder(
            @NonNull ViewGroup parent,
            int viewType
    ) {
        View view = LayoutInflater.from(context)
                .inflate(
                        R.layout.item_recept,
                        parent,
                        false
                );

        return new ReceptViewHolder(view);
    }

    @Override
    public void onBindViewHolder(
            @NonNull ReceptViewHolder holder,
            int position
    ) {
        Recept recept = receptList.get(position);

        String patientName = cleanText(
                recept.getPatientName()
        );

        if (patientName.isEmpty()) {
            patientName = recept.isMedicineHistory()
                    ? "Gyógyszerszkennelés"
                    : "Beteg neve nincs megadva";
        }

        String medicineName = cleanText(
                recept.getName()
        );

        if (medicineName.isEmpty()) {
            medicineName = "Ismeretlen gyógyszer";
        }

        String dosage = cleanText(
                recept.getPortion()
        );

        holder.patientNameTv.setText(patientName);
        holder.nameTv.setText(medicineName);
        holder.detailsTv.setText(
                dosage.isEmpty()
                        ? "Adagolás nincs megadva"
                        : "Adagolás: " + dosage
        );

        holder.dateTv.setText(
                recept.getTimestamp() == null
                        ? ""
                        : recept.getTimestamp()
        );

        String imageUrl = recept.getImageUrl();

        if (imageUrl != null
                && !imageUrl.trim().isEmpty()) {
            Glide.with(context)
                    .load(Uri.parse(imageUrl))
                    .placeholder(
                            android.R.drawable.ic_menu_gallery
                    )
                    .error(
                            android.R.drawable.ic_menu_report_image
                    )
                    .into(holder.imageView);
        } else {
            holder.imageView.setImageResource(
                    android.R.drawable.ic_menu_gallery
            );
        }

        String documentId = cleanText(
                recept.getDocumentId()
        );

        boolean selected =
                !documentId.isEmpty()
                        && selectedDocumentIds.contains(documentId);

        holder.selectionCheckBox.setVisibility(
                selectionMode
                        ? View.VISIBLE
                        : View.GONE
        );

        holder.selectionCheckBox.setChecked(selected);
        holder.cardView.setStrokeWidth(
                selected
                        ? dpToPx(3)
                        : dpToPx(1)
        );

        holder.cardView.setStrokeColor(
                selected
                        ? Color.parseColor("#6750A4")
                        : Color.parseColor("#79747E")
        );

        holder.cardView.setAlpha(
                selected
                        ? 0.88f
                        : 1.0f
        );

        holder.itemView.setOnLongClickListener(v -> {
            if (documentId.isEmpty()) {
                Toast.makeText(
                        context,
                        "Ez az előzmény nem jelölhető ki.",
                        Toast.LENGTH_SHORT
                ).show();
                return true;
            }

            if (!selectionMode) {
                selectionMode = true;
            }

            toggleSelection(documentId);
            return true;
        });

        holder.itemView.setOnClickListener(v -> {
            if (selectionMode) {
                toggleSelection(documentId);
            }
        });

        holder.selectionCheckBox.setOnClickListener(v ->
                toggleSelection(documentId)
        );

        holder.imageView.setOnClickListener(v -> {
            if (selectionMode) {
                toggleSelection(documentId);
                return;
            }

            if (imageUrl == null
                    || imageUrl.trim().isEmpty()) {
                Toast.makeText(
                        context,
                        "Ehhez az előzményhez nincs elmentett kép.",
                        Toast.LENGTH_SHORT
                ).show();
                return;
            }

            Intent intent =
                    new Intent(
                            context,
                            ImageDetailActivity.class
                    );

            intent.putExtra(
                    "image_uri",
                    imageUrl
            );

            context.startActivity(intent);
        });
    }

    private void toggleSelection(String documentId) {
        if (documentId == null
                || documentId.trim().isEmpty()) {
            return;
        }

        if (selectedDocumentIds.contains(documentId)) {
            selectedDocumentIds.remove(documentId);
        } else {
            selectedDocumentIds.add(documentId);
        }

        if (selectedDocumentIds.isEmpty()) {
            selectionMode = false;
        }

        notifyDataSetChanged();
        notifySelectionChanged();
    }

    public void clearSelection() {
        selectedDocumentIds.clear();
        selectionMode = false;
        notifyDataSetChanged();
        notifySelectionChanged();
    }

    public void pruneSelections() {
        Set<String> existingIds = new HashSet<>();

        for (Recept recept : receptList) {
            String documentId = cleanText(
                    recept.getDocumentId()
            );

            if (!documentId.isEmpty()) {
                existingIds.add(documentId);
            }
        }

        Iterator<String> iterator =
                selectedDocumentIds.iterator();

        while (iterator.hasNext()) {
            if (!existingIds.contains(iterator.next())) {
                iterator.remove();
            }
        }

        if (selectedDocumentIds.isEmpty()) {
            selectionMode = false;
        }

        notifySelectionChanged();
    }

    public List<Recept> getSelectedItems() {
        List<Recept> selectedItems =
                new ArrayList<>();

        for (Recept recept : receptList) {
            String documentId = cleanText(
                    recept.getDocumentId()
            );

            if (selectedDocumentIds.contains(documentId)) {
                selectedItems.add(recept);
            }
        }

        return selectedItems;
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    public int getSelectedCount() {
        return selectedDocumentIds.size();
    }

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(
                    selectedDocumentIds.size(),
                    selectionMode
            );
        }
    }

    private String cleanText(String value) {
        return value == null
                ? ""
                : value.trim();
    }

    private int dpToPx(int dp) {
        return Math.round(
                dp * context.getResources()
                        .getDisplayMetrics().density
        );
    }

    @Override
    public int getItemCount() {
        return receptList.size();
    }

    public static class ReceptViewHolder
            extends RecyclerView.ViewHolder {

        MaterialCardView cardView;
        MaterialCheckBox selectionCheckBox;
        ImageView imageView;
        TextView patientNameTv;
        TextView nameTv;
        TextView detailsTv;
        TextView dateTv;

        public ReceptViewHolder(
                @NonNull View itemView
        ) {
            super(itemView);

            cardView =
                    itemView.findViewById(
                            R.id.historyItemCard
                    );

            selectionCheckBox =
                    itemView.findViewById(
                            R.id.itemSelectionCheckBox
                    );

            imageView =
                    itemView.findViewById(
                            R.id.itemImageView
                    );

            patientNameTv =
                    itemView.findViewById(
                            R.id.itemPatientNameTextView
                    );

            nameTv =
                    itemView.findViewById(
                            R.id.itemNameTextView
                    );

            detailsTv =
                    itemView.findViewById(
                            R.id.itemDetailsTextView
                    );

            dateTv =
                    itemView.findViewById(
                            R.id.itemDateTextView
                    );
        }
    }
}
