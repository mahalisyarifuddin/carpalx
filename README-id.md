# Pengoptimal Tata Letak Keyboard Carpalx

**Carpalx** adalah pengoptimal tata letak keyboard yang menggunakan simulated annealing untuk meminimalkan upaya pengetikan. Repositori ini menawarkan tiga implementasi: aplikasi web mandiri, buku catatan Google Colab, dan skrip Python.

## Implementasi

### 1. Aplikasi Web (`carpalx.html`)

Alat HTML-JS mandiri untuk optimasi instan di browser Anda.

*   **Visualisasi Keyboard**: Perenderan tata letak secara real-time.
*   **Analisis Instan**: Hitung upaya untuk teks apa pun secara langsung.
*   **Optimasi**: Jalankan simulated annealing langsung di browser.

**Cara penggunaan:** Buka `carpalx.html` di browser web modern mana pun.

### 2. Buku Catatan Google Colab (`carpalx.ipynb`)

Buku catatan berbasis Python yang mandiri untuk eksperimen tingkat lanjut.

*   **Mandiri**: Semua model konfigurasi dan logika tertanam di dalamnya.
*   **Korpus Kustom**: Unggah dan analisis file teks Anda sendiri.
*   **Visualisasi Kaya**: Menggunakan Matplotlib untuk plot tata letak.

**Cara penggunaan:** Buka `carpalx.ipynb` di Google Colab atau lingkungan Jupyter lokal.

### 3. Skrip Python (`carpalx.py`)

Port Python inti dari logika asli Carpalx.

```bash
python3 carpalx.py -conf etc/carpalx.conf
```

## Implementasi Perl Lama

Implementasi Perl asli tersedia di direktori `legacy/` sebagai referensi.

## Dokumentasi Asli

Lihat [mkweb.bcgsc.ca/carpalx](http://mkweb.bcgsc.ca/carpalx) untuk dokumentasi proyek asli dan teori di balik model upaya pengetikan.

## Lisensi

GNU General Public License. Lihat file sumber untuk detailnya.
