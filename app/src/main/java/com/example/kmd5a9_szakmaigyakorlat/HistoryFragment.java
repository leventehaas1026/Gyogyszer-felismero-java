package com.example.kmd5a9_szakmaigyakorlat;

import android.app.AlertDialog;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButton;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.ListenerRegistration;
import com.google.firebase.firestore.QueryDocumentSnapshot;
import com.google.firebase.firestore.WriteBatch;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class HistoryFragment extends Fragment {

    private static final int DELETE_BATCH_SIZE = 400;

    private RecyclerView recyclerView;
    private ReceptAdapter adapter;
    private List<Recept> receptList;
    private FirebaseFirestore db;
    private TextView tvEmptyState;
    private TextView tvHistoryHint;
    private MaterialButton btnDeleteAllHistory;
    private MaterialButton btnCancelSelection;
    private ListenerRegistration historyListener;
    private boolean isDeletingHistory;

    @Nullable
    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater,
            @Nullable ViewGroup container,
            Bundle savedInstanceState
    ) {
        View view = inflater.inflate(
                R.layout.fragment_history,
                container,
                false
        );

        recyclerView = view.findViewById(R.id.historyRecyclerView);
        tvEmptyState = view.findViewById(R.id.tvEmptyState);
        tvHistoryHint = view.findViewById(R.id.tvHistoryHint);
        btnDeleteAllHistory = view.findViewById(R.id.btnDeleteAllHistory);
        btnCancelSelection = view.findViewById(R.id.btnCancelSelection);

        recyclerView.setLayoutManager(
                new LinearLayoutManager(requireContext())
        );

        receptList = new ArrayList<>();

        adapter = new ReceptAdapter(
                requireContext(),
                receptList,
                this::updateSelectionState
        );

        recyclerView.setAdapter(adapter);
        db = FirebaseFirestore.getInstance();

        btnDeleteAllHistory.setEnabled(false);
        btnDeleteAllHistory.setOnClickListener(v -> {
            if (adapter != null && adapter.isSelectionMode()) {
                showDeleteSelectedConfirmation();
            } else {
                showDeleteAllConfirmation();
            }
        });

        btnCancelSelection.setOnClickListener(v -> {
            if (adapter != null) {
                adapter.clearSelection();
            }
        });

        attachSwipeToDelete();
        loadDataFromFirestore();
        updateSelectionState(0, false);

        return view;
    }

    private void attachSwipeToDelete() {
        Paint backgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        backgroundPaint.setColor(Color.parseColor("#B3261E"));

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(
                16f * getResources().getDisplayMetrics().scaledDensity
        );
        textPaint.setTextAlign(Paint.Align.CENTER);

        ItemTouchHelper.SimpleCallback callback =
                new ItemTouchHelper.SimpleCallback(
                        0,
                        ItemTouchHelper.LEFT
                ) {
                    @Override
                    public boolean onMove(
                            @NonNull RecyclerView recyclerView,
                            @NonNull RecyclerView.ViewHolder viewHolder,
                            @NonNull RecyclerView.ViewHolder target
                    ) {
                        return false;
                    }

                    @Override
                    public int getSwipeDirs(
                            @NonNull RecyclerView recyclerView,
                            @NonNull RecyclerView.ViewHolder viewHolder
                    ) {
                        if (isDeletingHistory
                                || adapter == null
                                || adapter.isSelectionMode()) {
                            return 0;
                        }

                        return super.getSwipeDirs(
                                recyclerView,
                                viewHolder
                        );
                    }

                    @Override
                    public void onChildDraw(
                            @NonNull Canvas canvas,
                            @NonNull RecyclerView recyclerView,
                            @NonNull RecyclerView.ViewHolder viewHolder,
                            float dX,
                            float dY,
                            int actionState,
                            boolean isCurrentlyActive
                    ) {
                        if (dX < 0f) {
                            View itemView = viewHolder.itemView;

                            RectF background = new RectF(
                                    itemView.getRight() + dX,
                                    itemView.getTop(),
                                    itemView.getRight(),
                                    itemView.getBottom()
                            );

                            canvas.drawRoundRect(
                                    background,
                                    18f,
                                    18f,
                                    backgroundPaint
                            );

                            float centerX =
                                    itemView.getRight()
                                            - Math.min(
                                            Math.abs(dX) / 2f,
                                            72f * getResources()
                                                    .getDisplayMetrics().density
                                    );

                            float centerY =
                                    itemView.getTop()
                                            + itemView.getHeight() / 2f
                                            - (
                                            textPaint.descent()
                                                    + textPaint.ascent()
                                    ) / 2f;

                            canvas.drawText(
                                    "Törlés",
                                    centerX,
                                    centerY,
                                    textPaint
                            );
                        }

                        super.onChildDraw(
                                canvas,
                                recyclerView,
                                viewHolder,
                                dX,
                                dY,
                                actionState,
                                isCurrentlyActive
                        );
                    }

                    @Override
                    public void onSwiped(
                            @NonNull RecyclerView.ViewHolder viewHolder,
                            int direction
                    ) {
                        int position =
                                viewHolder.getBindingAdapterPosition();

                        if (position == RecyclerView.NO_POSITION
                                || position >= receptList.size()) {
                            adapter.notifyDataSetChanged();
                            return;
                        }

                        Recept item = receptList.get(position);
                        showDeleteSingleConfirmation(item, position);
                    }
                };

        new ItemTouchHelper(callback)
                .attachToRecyclerView(recyclerView);
    }

    private void showDeleteSingleConfirmation(
            Recept item,
            int position
    ) {
        new AlertDialog.Builder(requireContext())
                .setTitle("Előzmény törlése")
                .setMessage(
                        "Biztosan törölni szeretnéd ezt az előzményt?"
                )
                .setNegativeButton(
                        "Mégse",
                        (dialog, which) ->
                                adapter.notifyItemChanged(position)
                )
                .setPositiveButton(
                        "Törlés",
                        (dialog, which) ->
                                deleteSingleHistory(item, position)
                )
                .setOnCancelListener(dialog ->
                        adapter.notifyItemChanged(position)
                )
                .show();
    }

    private void showDeleteAllConfirmation() {
        if (isDeletingHistory) {
            return;
        }

        if (receptList == null || receptList.isEmpty()) {
            Toast.makeText(
                    requireContext(),
                    "Nincs törölhető előzmény.",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        new AlertDialog.Builder(requireContext())
                .setTitle("Minden előzmény törlése")
                .setMessage(
                        "Biztosan törölni szeretnéd az összes előzményt?\n\n"
                                + "Törlendő elemek száma: "
                                + receptList.size()
                                + "\n\nEz a művelet nem vonható vissza."
                )
                .setNegativeButton("Mégse", null)
                .setPositiveButton(
                        "Minden törlése",
                        (dialog, which) -> deleteAllUserHistory()
                )
                .show();
    }

    private void showDeleteSelectedConfirmation() {
        if (isDeletingHistory || adapter == null) {
            return;
        }

        List<Recept> selectedItems =
                adapter.getSelectedItems();

        if (selectedItems.isEmpty()) {
            Toast.makeText(
                    requireContext(),
                    "Jelölj ki legalább egy előzményt.",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        new AlertDialog.Builder(requireContext())
                .setTitle("Kijelölt előzmények törlése")
                .setMessage(
                        "Biztosan törölni szeretnéd a kijelölt előzményeket?\n\n"
                                + "Törlendő elemek száma: "
                                + selectedItems.size()
                                + "\n\nEz a művelet nem vonható vissza."
                )
                .setNegativeButton("Mégse", null)
                .setPositiveButton(
                        "Kijelöltek törlése",
                        (dialog, which) ->
                                deleteSelectedHistory(selectedItems)
                )
                .show();
    }

    private void deleteSingleHistory(
            Recept item,
            int swipedPosition
    ) {
        if (isDeletingHistory) {
            adapter.notifyItemChanged(swipedPosition);
            return;
        }

        if (item == null
                || item.getDocumentId() == null
                || item.getDocumentId().trim().isEmpty()) {
            Toast.makeText(
                    requireContext(),
                    "Az előzmény azonosítója hiányzik.",
                    Toast.LENGTH_SHORT
            ).show();
            adapter.notifyItemChanged(swipedPosition);
            return;
        }

        isDeletingHistory = true;
        updateSelectionState(0, false);

        db.collection("receptek")
                .document(item.getDocumentId())
                .delete()
                .addOnSuccessListener(unused -> {
                    deleteLocalImage(item.getImageUrl());
                    isDeletingHistory = false;
                    updateEmptyState();

                    if (isAdded()) {
                        Toast.makeText(
                                requireContext(),
                                "Előzmény törölve.",
                                Toast.LENGTH_SHORT
                        ).show();
                    }
                })
                .addOnFailureListener(error -> {
                    Log.e(
                            "HistoryDelete",
                            "Az előzmény törlése sikertelen.",
                            error
                    );

                    isDeletingHistory = false;
                    adapter.notifyItemChanged(swipedPosition);
                    updateSelectionState(
                            adapter.getSelectedCount(),
                            adapter.isSelectionMode()
                    );

                    if (isAdded()) {
                        Toast.makeText(
                                requireContext(),
                                "Nem sikerült törölni az előzményt.",
                                Toast.LENGTH_SHORT
                        ).show();
                    }
                });
    }

    private void deleteSelectedHistory(
            List<Recept> selectedItems
    ) {
        if (selectedItems == null
                || selectedItems.isEmpty()
                || isDeletingHistory) {
            return;
        }

        isDeletingHistory = true;
        updateSelectionState(
                adapter.getSelectedCount(),
                true
        );

        deleteSelectedBatch(
                selectedItems,
                0,
                new ArrayList<>()
        );
    }

    private void deleteSelectedBatch(
            List<Recept> selectedItems,
            int startIndex,
            List<String> localImageUrls
    ) {
        if (startIndex >= selectedItems.size()) {
            for (String imageUrl : localImageUrls) {
                deleteLocalImage(imageUrl);
            }

            isDeletingHistory = false;
            adapter.clearSelection();
            updateEmptyState();

            if (isAdded()) {
                Toast.makeText(
                        requireContext(),
                        "A kijelölt előzmények törölve.",
                        Toast.LENGTH_SHORT
                ).show();
            }

            return;
        }

        int endIndex = Math.min(
                startIndex + DELETE_BATCH_SIZE,
                selectedItems.size()
        );

        WriteBatch batch = db.batch();

        for (int index = startIndex;
             index < endIndex;
             index++) {
            Recept item = selectedItems.get(index);

            if (item.getDocumentId() != null
                    && !item.getDocumentId().trim().isEmpty()) {
                batch.delete(
                        db.collection("receptek")
                                .document(item.getDocumentId())
                );
            }

            if (item.getImageUrl() != null
                    && !item.getImageUrl().trim().isEmpty()) {
                localImageUrls.add(item.getImageUrl());
            }
        }

        batch.commit()
                .addOnSuccessListener(unused ->
                        deleteSelectedBatch(
                                selectedItems,
                                endIndex,
                                localImageUrls
                        )
                )
                .addOnFailureListener(error -> {
                    Log.e(
                            "HistoryDelete",
                            "A kijelölt előzmények törlése sikertelen.",
                            error
                    );

                    isDeletingHistory = false;
                    updateSelectionState(
                            adapter.getSelectedCount(),
                            true
                    );

                    if (isAdded()) {
                        Toast.makeText(
                                requireContext(),
                                "A kijelölt előzmények törlése nem sikerült.",
                                Toast.LENGTH_LONG
                        ).show();
                    }
                });
    }

    private void deleteAllUserHistory() {
        FirebaseUser currentUser =
                FirebaseAuth.getInstance().getCurrentUser();

        if (currentUser == null) {
            Toast.makeText(
                    requireContext(),
                    "Az előzmények törléséhez be kell jelentkezni.",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        if (isDeletingHistory) {
            return;
        }

        isDeletingHistory = true;
        updateSelectionState(0, false);

        db.collection("receptek")
                .whereEqualTo(
                        "userId",
                        currentUser.getUid()
                )
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    if (!isAdded()) {
                        isDeletingHistory = false;
                        return;
                    }

                    List<QueryDocumentSnapshot> documents =
                            new ArrayList<>();

                    List<String> localImageUrls =
                            new ArrayList<>();

                    for (QueryDocumentSnapshot document
                            : querySnapshot) {
                        documents.add(document);

                        Recept recept =
                                document.toObject(Recept.class);

                        if (recept != null
                                && recept.getImageUrl() != null
                                && !recept.getImageUrl().trim().isEmpty()) {
                            localImageUrls.add(
                                    recept.getImageUrl()
                            );
                        }
                    }

                    if (documents.isEmpty()) {
                        isDeletingHistory = false;
                        updateEmptyState();

                        Toast.makeText(
                                requireContext(),
                                "Nincs törölhető előzmény.",
                                Toast.LENGTH_SHORT
                        ).show();
                        return;
                    }

                    deleteAllHistoryBatch(
                            documents,
                            localImageUrls,
                            0
                    );
                })
                .addOnFailureListener(error -> {
                    Log.e(
                            "HistoryDelete",
                            "Az előzmények lekérése sikertelen.",
                            error
                    );

                    isDeletingHistory = false;
                    updateEmptyState();

                    if (isAdded()) {
                        Toast.makeText(
                                requireContext(),
                                "Nem sikerült lekérni az előzményeket.",
                                Toast.LENGTH_SHORT
                        ).show();
                    }
                });
    }

    private void deleteAllHistoryBatch(
            List<QueryDocumentSnapshot> documents,
            List<String> localImageUrls,
            int startIndex
    ) {
        if (startIndex >= documents.size()) {
            for (String imageUrl : localImageUrls) {
                deleteLocalImage(imageUrl);
            }

            isDeletingHistory = false;

            if (adapter != null) {
                adapter.clearSelection();
            }

            updateEmptyState();

            if (isAdded()) {
                Toast.makeText(
                        requireContext(),
                        "Minden előzmény törölve.",
                        Toast.LENGTH_SHORT
                ).show();
            }

            return;
        }

        int endIndex = Math.min(
                startIndex + DELETE_BATCH_SIZE,
                documents.size()
        );

        WriteBatch batch = db.batch();

        for (int index = startIndex;
             index < endIndex;
             index++) {
            batch.delete(
                    documents.get(index).getReference()
            );
        }

        batch.commit()
                .addOnSuccessListener(unused ->
                        deleteAllHistoryBatch(
                                documents,
                                localImageUrls,
                                endIndex
                        )
                )
                .addOnFailureListener(error -> {
                    Log.e(
                            "HistoryDelete",
                            "Minden előzmény törlése sikertelen.",
                            error
                    );

                    isDeletingHistory = false;
                    updateEmptyState();

                    if (isAdded()) {
                        Toast.makeText(
                                requireContext(),
                                "Nem sikerült minden előzményt törölni.",
                                Toast.LENGTH_LONG
                        ).show();
                    }
                });
    }

    private void deleteLocalImage(String imageUrl) {
        if (imageUrl == null || imageUrl.trim().isEmpty()) {
            return;
        }

        try {
            Uri uri = Uri.parse(imageUrl);

            if ("file".equalsIgnoreCase(uri.getScheme())
                    || uri.getScheme() == null) {
                String path = uri.getPath();

                if (path != null) {
                    File file = new File(path);

                    if (file.exists() && !file.delete()) {
                        Log.w(
                                "HistoryDelete",
                                "A helyi képfájl nem törölhető: " + path
                        );
                    }
                }
            }
        } catch (Exception error) {
            Log.e(
                    "HistoryDelete",
                    "A helyi képfájl törlése sikertelen.",
                    error
            );
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        loadDataFromFirestore();
    }

    private void loadDataFromFirestore() {
        FirebaseUser currentUser =
                FirebaseAuth.getInstance().getCurrentUser();

        if (currentUser == null) {
            receptList.clear();
            adapter.notifyDataSetChanged();
            adapter.clearSelection();
            updateEmptyState();
            return;
        }

        if (historyListener != null) {
            historyListener.remove();
            historyListener = null;
        }

        historyListener =
                db.collection("receptek")
                        .whereEqualTo(
                                "userId",
                                currentUser.getUid()
                        )
                        .orderBy(
                                "timestamp",
                                com.google.firebase.firestore.Query
                                        .Direction.DESCENDING
                        )
                        .addSnapshotListener(
                                (queryDocumentSnapshots, error) -> {
                                    if (error != null) {
                                        Log.e(
                                                "FirestoreError",
                                                "Az előzmények betöltése sikertelen.",
                                                error
                                        );
                                        return;
                                    }

                                    if (queryDocumentSnapshots == null
                                            || receptList == null
                                            || adapter == null) {
                                        return;
                                    }

                                    receptList.clear();

                                    for (QueryDocumentSnapshot document
                                            : queryDocumentSnapshots) {
                                        Recept recept =
                                                document.toObject(
                                                        Recept.class
                                                );

                                        recept.setDocumentId(
                                                document.getId()
                                        );

                                        receptList.add(recept);
                                    }

                                    adapter.pruneSelections();
                                    adapter.notifyDataSetChanged();
                                    updateEmptyState();
                                }
                        );
    }

    private void updateEmptyState() {
        boolean isEmpty =
                receptList == null
                        || receptList.isEmpty();

        if (tvEmptyState != null) {
            tvEmptyState.setVisibility(
                    isEmpty
                            ? View.VISIBLE
                            : View.GONE
            );
        }

        if (recyclerView != null) {
            recyclerView.setVisibility(
                    isEmpty
                            ? View.GONE
                            : View.VISIBLE
            );
        }

        if (isEmpty && adapter != null) {
            adapter.clearSelection();
        }

        updateSelectionState(
                adapter == null
                        ? 0
                        : adapter.getSelectedCount(),
                adapter != null
                        && adapter.isSelectionMode()
        );
    }

    private void updateSelectionState(
            int selectedCount,
            boolean selectionMode
    ) {
        if (btnDeleteAllHistory == null
                || btnCancelSelection == null
                || tvHistoryHint == null) {
            return;
        }

        boolean hasItems =
                receptList != null
                        && !receptList.isEmpty();

        if (selectionMode) {
            btnDeleteAllHistory.setText(
                    isDeletingHistory
                            ? "Törlés folyamatban..."
                            : "Kijelöltek törlése (" + selectedCount + ")"
            );

            btnDeleteAllHistory.setBackgroundTintList(
                    ColorStateList.valueOf(
                            Color.parseColor("#B3261E")
                    )
            );

            btnDeleteAllHistory.setEnabled(
                    selectedCount > 0
                            && !isDeletingHistory
            );

            btnCancelSelection.setVisibility(View.VISIBLE);
            btnCancelSelection.setEnabled(!isDeletingHistory);

            tvHistoryHint.setText(
                    "Koppints a jelölőnégyzetekre. "
                            + "A kijelölés befejezéséhez nyomd meg a Mégse gombot."
            );
        } else {
            btnDeleteAllHistory.setText(
                    isDeletingHistory
                            ? "Törlés folyamatban..."
                            : "Minden előzmény törlése"
            );

            btnDeleteAllHistory.setBackgroundTintList(
                    ColorStateList.valueOf(
                            Color.parseColor("#6750A4")
                    )
            );

            btnDeleteAllHistory.setEnabled(
                    hasItems
                            && !isDeletingHistory
            );

            btnCancelSelection.setVisibility(View.GONE);

            tvHistoryHint.setText(
                    "Egy elem törléséhez húzd balra. "
                            + "Több elem kijelöléséhez tartsd hosszan."
            );
        }
    }

    @Override
    public void onDestroyView() {
        if (historyListener != null) {
            historyListener.remove();
            historyListener = null;
        }

        recyclerView = null;
        adapter = null;
        receptList = null;
        tvEmptyState = null;
        tvHistoryHint = null;
        btnDeleteAllHistory = null;
        btnCancelSelection = null;

        super.onDestroyView();
    }
}
