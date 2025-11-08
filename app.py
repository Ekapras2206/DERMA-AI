# === MODE 2: CAMERA SNAPSHOT ===
elif mode == "Ambil dari Kamera":
    camera_img = st.camera_input("📷 Ambil foto dari kamera")

    if camera_img is not None:
        try:
            img = Image.open(camera_img).convert("RGB")
            st.image(img, caption="Foto hasil kamera", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal membaca gambar dari kamera: {e}")
    else:
        st.info("Silakan ambil foto terlebih dahulu.")

# === MODE 3: LIVE SCAN (SIMULATED LOOP) ===
elif mode == "Live Scan":
    st.warning("🟢 Mode Live Scan (simulasi) — cocok untuk kamera HP.")

    if "scan_active" not in st.session_state:
        st.session_state.scan_active = False

    start_btn = st.button("▶️ Mulai Scan")
    stop_btn = st.button("⏹️ Berhenti")

    if start_btn:
        st.session_state.scan_active = True
    if stop_btn:
        st.session_state.scan_active = False

    if st.session_state.scan_active:
        live_img = st.camera_input("Ambil frame untuk dianalisis")
        if live_img is not None:
            try:
                img = Image.open(live_img).convert("RGB")
                st.image(img, caption="Frame terbaru", use_container_width=True)

                # Prediksi langsung
                img_tensor = preprocess_pil(img)
                label, conf = get_prediction(img_tensor)
                st.markdown(
                    f"<h3 style='text-align:center;'>Prediksi: {label.upper()} — {conf*100:.2f}%</h3>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Error saat memproses frame: {e}")
        else:
            st.info("📸 Arahkan kamera ke kulit dan ambil gambar untuk melihat prediksi.")
