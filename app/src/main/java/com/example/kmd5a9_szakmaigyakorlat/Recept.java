package com.example.kmd5a9_szakmaigyakorlat;

public class Recept {

    public static final String TYPE_PRESCRIPTION = "prescription";
    public static final String TYPE_MEDICINE = "medicine";

    private String name;
    private int amount;
    private String portion;
    private String timestamp;
    private String weight;
    private String weightUnit;
    private String imageUrl;
    private String userId;
    private String documentId;
    private String patientName;
    private String historyType;
    private double similarity;
    private String recognitionSide;

    public Recept() {
    }

    public Recept(
            String name,
            int amount,
            String portion,
            String timestamp,
            String weight,
            String weightUnit,
            String imageUrl,
            String userId
    ) {
        this.name = name;
        this.amount = amount;
        this.portion = portion;
        this.timestamp = timestamp;
        this.weight = weight;
        this.weightUnit = weightUnit;
        this.imageUrl = imageUrl;
        this.userId = userId;
        this.historyType = TYPE_PRESCRIPTION;
    }

    public boolean isMedicineHistory() {
        return TYPE_MEDICINE.equalsIgnoreCase(historyType);
    }

    public boolean isPrescriptionHistory() {
        return historyType == null
                || historyType.trim().isEmpty()
                || TYPE_PRESCRIPTION.equalsIgnoreCase(historyType);
    }

    public String getPatientName() {
        return patientName;
    }

    public void setPatientName(String patientName) {
        this.patientName = patientName;
    }

    public String getDocumentId() {
        return documentId;
    }

    public void setDocumentId(String documentId) {
        this.documentId = documentId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getWeightUnit() {
        return weightUnit;
    }

    public void setWeightUnit(String weightUnit) {
        this.weightUnit = weightUnit;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAmount() {
        return amount;
    }

    public void setAmount(int amount) {
        this.amount = amount;
    }

    public String getPortion() {
        return portion;
    }

    public void setPortion(String portion) {
        this.portion = portion;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getWeight() {
        return weight;
    }

    public void setWeight(String weight) {
        this.weight = weight;
    }

    public String getHistoryType() {
        return historyType;
    }

    public void setHistoryType(String historyType) {
        this.historyType = historyType;
    }

    public double getSimilarity() {
        return similarity;
    }

    public void setSimilarity(double similarity) {
        this.similarity = similarity;
    }

    public String getRecognitionSide() {
        return recognitionSide;
    }

    public void setRecognitionSide(String recognitionSide) {
        this.recognitionSide = recognitionSide;
    }
}
