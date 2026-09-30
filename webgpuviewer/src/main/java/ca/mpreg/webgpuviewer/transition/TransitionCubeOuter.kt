package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage
import kotlin.math.cos
import kotlin.math.sin

object TransitionCubeOuter : Transition() {
    private const val HALF_PI = (Math.PI / 2.0).toFloat()
    private const val FOV = 4f
    private const val FACE_DEPTH = FOV / (FOV - 1f)

    private fun mat4(
        m00: Float, m01: Float, m02: Float, m03: Float,
        m10: Float, m11: Float, m12: Float, m13: Float,
        m20: Float, m21: Float, m22: Float, m23: Float,
        m30: Float, m31: Float, m32: Float, m33: Float,
    ) = floatArrayOf(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    )

    private fun rotateY(angle: Float): FloatArray {
        val c = cos(angle)
        val s = sin(angle)
        return mat4(
            c, 0f, -s, 0f,
            0f, 1f, 0f, 0f,
            s, 0f, c, 0f,
            0f, 0f, 0f, 1f,
        )
    }

    private fun translate(x: Float, y: Float, z: Float) = mat4(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        x, y, z, 1f,
    )

    private fun scale(x: Float, y: Float, z: Float) = mat4(
        x, 0f, 0f, 0f,
        0f, y, 0f, 0f,
        0f, 0f, z, 0f,
        0f, 0f, 0f, 1f,
    )

    private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val result = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                result[col * 4 + row] = sum
            }
        }
        return result
    }

    /**
     * Face transform for one side of the cube.
     *
     * [faceWidth] and [faceHeight] size the face so its content isn't distorted. This shares
     * [TransitionCube]'s shader, whose unit quad is the whole cached surface, so they are the
     * surface's dimensions - the vertical scale then cancels against [screenAspect] and the face
     * comes out screen-shaped.
     */
    private fun buildFaceMatrix(
        rotAngle: Float,
        screenAspect: Float,
        faceWidth: Float,
        faceHeight: Float,
        isSide: Boolean,
    ): FloatArray {
        val d = FACE_DEPTH
        val pushBack = 5f * d
        // Scale face so it fills NDC ±1 at rest, matching the flat page size
        // ndc edge = s * FOV / (pushBack - s + FOV) = 1  →  s = (pushBack + FOV) / (FOV + 1)
        val s = (pushBack + FOV) / (FOV + 1f)
        val faceScaleMat = scale(s, (faceHeight / faceWidth) * screenAspect * s, 1f)
        val baseMat = if (isSide) {
            multiply(rotateY(HALF_PI), multiply(translate(0f, 0f, -s), faceScaleMat))
        } else {
            multiply(translate(0f, 0f, -s), faceScaleMat)
        }
        val mat = multiply(translate(0f, 0f, pushBack), multiply(rotateY(-rotAngle), baseMat))
        // Perspective projection: output.w = z + fov, ndc = xy / w
        val projMat = mat4(
            FOV, 0f, 0f, 0f,
            0f, FOV, 0f, 0f,
            0f, 0f, 1f, 1f,
            0f, 0f, 0f, FOV,
        )
        return multiply(projMat, mat)
    }

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        // CubeOuter rotates opposite to Cube, hence negated frac
        val t = if (frac < 0f) -frac else 1f - frac

        val rotAngle = t * HALF_PI

        val screenAspect = dst.width.toFloat() / dst.height.toFloat()

        // Negated frac means page selection is swapped vs TransitionCube
        val frontPage: ImagePage
        val sidePage: ImagePage
        val frontIsPage1: Boolean
        if (frac < 0f) {
            frontPage = page1
            sidePage = page2
            frontIsPage1 = true
        } else {
            frontPage = page2
            sidePage = page1
            frontIsPage1 = false
        }

        // A face is the whole cached surface, so both use the surface's dimensions.
        val faceWidth = dst.width.toFloat()
        val faceHeight = dst.height.toFloat()

        val frontMat = buildFaceMatrix(
            rotAngle, screenAspect, faceWidth, faceHeight, isSide = false,
        )
        val sideMat = buildFaceMatrix(
            rotAngle, screenAspect, faceWidth, faceHeight, isSide = true,
        )

        // The cube never covers the whole surface, and getCurrentTexture hands back a rotating set
        // of buffers, so without a clear the area around it shows a frame from several ago.
        Draw.clear(encoder, dst, 0)

        if (t < 0.5f) {
            TransitionCube.face(sidePage, !frontIsPage1, encoder, dst, sideMat, tiles)
            TransitionCube.face(frontPage, frontIsPage1, encoder, dst, frontMat, tiles)
        } else {
            TransitionCube.face(frontPage, frontIsPage1, encoder, dst, frontMat, tiles)
            TransitionCube.face(sidePage, !frontIsPage1, encoder, dst, sideMat, tiles)
        }
    }
}
