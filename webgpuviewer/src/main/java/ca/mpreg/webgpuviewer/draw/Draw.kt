package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoder
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.submitAndRelease

object Draw {
    internal val device get() = WebGpuRenderer.device

    private val tempBuffers = ThreadLocal.withInitial { mutableListOf<GPUBuffer>() }

    fun submit(block: Draw.(GPUCommandEncoder) -> Unit) {
        val buffers = tempBuffers.get()
        var encoder: GPUCommandEncoder? = device.createCommandEncoder()
        try {
            block.invoke(this, encoder!!)
            device.queue.submitAndRelease(encoder)
            encoder = null
        } finally {
            encoder?.close()
            buffers.forEach {
                it.destroy()
                it.close()
            }
            buffers.clear()
        }
    }

    internal fun createBuffer(size: Long, usage: Int): GPUBuffer {
        return device.createBuffer(GPUBufferDescriptor(size = size, usage = usage)).also {
            tempBuffers.get().add(it)
        }
    }
}
