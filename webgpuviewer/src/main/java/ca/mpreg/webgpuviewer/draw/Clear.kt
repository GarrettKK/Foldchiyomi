package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.endAndRelease

fun Draw.clear(encoder: GPUCommandEncoder, texture: GPUTexture, color: Int) {
    val r = ((color shr 16) and 0xFF) / 255.0
    val g = ((color shr 8) and 0xFF) / 255.0
    val b = (color and 0xFF) / 255.0
    val a = ((color ushr 24) and 0xFF) / 255.0

    val targetView = texture.createView()
    encoder.beginRenderPass(
        GPURenderPassDescriptor(
            colorAttachments = arrayOf(
                GPURenderPassColorAttachment(
                    view = targetView,
                    loadOp = LoadOp.Clear,
                    storeOp = StoreOp.Store,
                    clearValue = GPUColor(r, g, b, a)
                )
            )
        )
    ).endAndRelease(targetView)
}
