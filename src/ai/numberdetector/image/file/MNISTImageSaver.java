package ai.numberdetector.image.file;

import java.io.FileOutputStream;
import java.io.IOException;

public class MNISTImageSaver {

    public static void saveImageAsBMP(double[] pixels, int width, int height, String filename, int label) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            // BMP file header (14 bytes)
            int fileSize = 54 + width * height * 3; // 54 header + 3 bytes per pixel
            writeBMPHeader(fos, fileSize, width, height);

            // BMP info header (40 bytes)
            writeBMPInfoHeader(fos, width, height);

            // Pixel data (bottom-up, BGR format)
            writePixelData(fos, pixels, width, height);
        }
        System.out.println("Saved image as: " + filename + " (Label: " + label + ")");
    }

    private static void writeBMPHeader(FileOutputStream fos, int fileSize, int width, int height) throws IOException {
        // BMP signature
        fos.write('B');
        fos.write('M');

        // File size
        writeInt(fos, fileSize);

        // Reserved
        writeShort(fos, 0);
        writeShort(fos, 0);

        // Pixel data offset
        writeInt(fos, 54);
    }

    private static void writeBMPInfoHeader(FileOutputStream fos, int width, int height) throws IOException {
        // Header size
        writeInt(fos, 40);

        // Image dimensions
        writeInt(fos, width);
        writeInt(fos, height);

        // Planes
        writeShort(fos, 1);

        // Bits per pixel (24-bit RGB)
        writeShort(fos, 24);

        // Compression (none)
        writeInt(fos, 0);

        // Image size (can be 0 for uncompressed)
        writeInt(fos, 0);

        // Resolution (pixels per meter)
        writeInt(fos, 0);
        writeInt(fos, 0);

        // Colors in color palette
        writeInt(fos, 0);
        writeInt(fos, 0);
    }

    private static void writePixelData(FileOutputStream fos, double[] pixels, int width, int height) throws IOException {
        // BMP stores pixels bottom-up, so we start from the last row
        int rowPadding = (4 - (width * 3) % 4) % 4; // Each row must be multiple of 4 bytes

        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int pixelValue = (int) (pixels[index] * 255); // Convert back to 0-255

                // Write BGR (Blue, Green, Red)
                fos.write(pixelValue); // Blue
                fos.write(pixelValue); // Green
                fos.write(pixelValue); // Red
            }

            // Write row padding
            for (int p = 0; p < rowPadding; p++) {
                fos.write(0);
            }
        }
    }

    private static void writeShort(FileOutputStream fos, int value) throws IOException {
        fos.write(value & 0xFF);
        fos.write((value >> 8) & 0xFF);
    }

    private static void writeInt(FileOutputStream fos, int value) throws IOException {
        fos.write(value & 0xFF);
        fos.write((value >> 8) & 0xFF);
        fos.write((value >> 16) & 0xFF);
        fos.write((value >> 24) & 0xFF);
    }
}