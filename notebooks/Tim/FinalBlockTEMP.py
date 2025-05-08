# ╔══════════════════════════════════════════════════════════════════════╗
# 5 |  Final Training +  Evaluation auf Full‑Set                         #
# ╚══════════════════════════════════════════════════════════════════════╝
head("FINAL  MODEL  – FIT AUF ALLEN DATEN  &  END‑METRIKEN")

X_full = np.vstack([X_train, X_val])
y_full = np.hstack([y_train, y_val])

final_params = {**base_params, 'early_stopping_rounds': None}   # ①
final_model  = XGBClassifier(**final_params).fit(X_full, y_full)  # ②

final_iso   = CalibratedClassifierCV(final_model, method='isotonic', cv=5).fit(X_full, y_full)
full_scores = final_iso.predict_proba(X_full)[:,1]
full_pred   = (full_scores >= tau_star).astype(int)

# ---- Gesamte Metriken --------------------------------------------------
final_metrics = {
    'PR‑AUC'           : average_precision_score(y_full, full_scores),
    'Precision'        : precision_score(y_full, full_pred, zero_division=0),
    'Recall'           : recall_score(y_full, full_pred, zero_division=0),
    'F1'               : f1_score(y_full, full_pred, zero_division=0),
    'F2'               : fbeta_score(y_full, full_pred, beta=2, zero_division=0),
    'BalancedAccuracy' : balanced_accuracy_score(y_full, full_pred)
}
print("\n===  FULL 1985‑2018  PERFORMANCE  ===")
for k,v in final_metrics.items():
    print(f"{k:<18}: {v:.3f}")
logging.info(f"Full‑set metrics: {final_metrics}")

# ---- PR‑Kurve & Confusion‑Matrix Plot ---------------------------------
head("PLOTS")

fig,ax = plt.subplots(1,2,figsize=(12,5))

# PR‑Curve
P,R,_ = precision_recall_curve(y_full, full_scores)
ax[0].plot(R,P,label=f'PR‑AUC {final_metrics["PR‑AUC"]:.3f}')
ax[0].scatter(final_metrics['Recall'], final_metrics['Precision'],
              c='red', label=f'τ*={tau_star:.3f}')
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].legend()
ax[0].set_title('Precision‑Recall Gesamt')

# Confusion‑Matrix
cm = confusion_matrix(y_full, full_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax[1])
ax[1].set_xlabel('Pred'); ax[1].set_ylabel('True');
ax[1].set_title('Confusion Matrix\n[TN FP / FN TP]')

plt.tight_layout(); plt.show()

# ---- Feature Importance Plot (Top‑15) ----------------------------------
fi = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=fi.head(15), y=fi.head(15).index, orient='h')
plt.title("Top‑15 Feature Importance"); plt.tight_layout(); plt.show()