package ca.mpreg.webgpuviewer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.MapMode
import ca.mpreg.webgpuviewer.Trim.Companion.detectBackgroundInContext
import ca.mpreg.webgpuviewer.Trim.Companion.findInContext
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.Mipmap
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.renderer.submitAndRelease
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.time.Duration.Companion.milliseconds

class Trim {
    companion object {
        private const val TAG = "Trim"

        private val device get() = WebGpuRenderer.device
        private val instance get() = WebGpuRenderer.instance

        /**
         * CPU counterpart of [findInContext], returning one [Rect] per colour in input order.
         *
         * Reads the decoded pixels rather than an uploaded texture, so it needs no GPU round
         * trip and can run on any background dispatcher. All colours are resolved in a single
         * pass over the image instead of one dispatch each.
         *
         * [pixels] must be a direct buffer of tightly packed RGBA8, `width * height * 4` bytes.
         * Falls back to full bounds if the native pass can't read it.
         */
        fun findAllCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            colors: List<FloatArray>,
            threshold: Float
        ): List<Rect> {
            require(colors.isNotEmpty()) { "colors must not be empty" }
            require(colors.all { it.size >= 3 }) { "each color must have at least 3 elements [r, g, b]" }

            val flat = FloatArray(colors.size * 3)
            colors.forEachIndexed { i, color ->
                flat[i * 3] = color[0]
                flat[i * 3 + 1] = color[1]
                flat[i * 3 + 2] = color[2]
            }

            val bounds = IntArray(colors.size * 4)
            if (!TrimNative.findTrim(pixels, width, height, flat, threshold, bounds)) {
                Log.w(TAG, "findAllCpu: native trim rejected ${width}x$height, using full bounds")
                return List(colors.size) { Rect(0, 0, width, height) }
            }

            // Same max -> exclusive-edge conversion the GPU readback does.
            return List(colors.size) { i ->
                Rect(
                    bounds[i * 4],
                    bounds[i * 4 + 1],
                    (bounds[i * 4 + 2] + 1).coerceAtMost(width),
                    (bounds[i * 4 + 3] + 1).coerceAtMost(height),
                )
            }
        }

        /** CPU counterpart of [findInContext]: the tightest trim across [colors]. */
        fun findCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            colors: List<FloatArray>,
            threshold: Float
        ): Rect = findAllCpu(pixels, width, height, colors, threshold)
            .minByOrNull { it.width().toLong() * it.height() } ?: Rect(0, 0, width, height)

        /**
         * CPU counterpart of [detectBackgroundInContext]: the background colour implied by the
         * image edges, as 0xAARRGGBB, or opaque white when no edge is a solid colour.
         */
        fun detectBackgroundCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            threshold: Float = 0.05f
        ): Int {
            if (!pixels.isDirect) {
                Log.w(TAG, "detectBackgroundCpu: pixels not direct, defaulting to white")
                return 0xFFFFFFFF.toInt()
            }
            return TrimNative.detectBackground(pixels, width, height, threshold)
        }

        /**
         * Detect background color by checking if any edge has low variance (solid color).
         * Returns the detected color (0xAARRGGBB) or opaque white (0xFFFFFFFF) if no solid edge found.
         */
        suspend fun detectBackgroundInContext(
            image: Image,
            threshold: Float = 0.05f
        ): Int {
            if (image.mipmaps.isEmpty()) return 0xFFFFFFFF.toInt()
            val mipmap = image.mipmaps[0]
            if (mipmap.textures.isEmpty()) return 0xFFFFFFFF.toInt()

            return coroutineScope {
                // Collect results grouped by edge direction
                val leftResults = mutableListOf<Deferred<EdgeResult>>()
                val rightResults = mutableListOf<Deferred<EdgeResult>>()
                val topResults = mutableListOf<Deferred<EdgeResult>>()
                val bottomResults = mutableListOf<Deferred<EdgeResult>>()

                // Left edge: first column of tiles
                for (row in 0 until mipmap.tilesRows) {
                    val idx = row * mipmap.tilesCols
                    if (idx < mipmap.textures.size) {
                        leftResults.add(
                            dispatchEdgeDetect(
                                mipmap.textures[idx],
                                Edge.LEFT,
                                threshold
                            )
                        )
                    }
                }

                // Right edge: last column of tiles
                for (row in 0 until mipmap.tilesRows) {
                    val idx = row * mipmap.tilesCols + mipmap.tilesCols - 1
                    if (idx < mipmap.textures.size) {
                        rightResults.add(
                            dispatchEdgeDetect(
                                mipmap.textures[idx],
                                Edge.RIGHT,
                                threshold
                            )
                        )
                    }
                }

                // Top edge: first row of tiles
                for (col in 0 until mipmap.tilesCols) {
                    val idx = col
                    if (idx < mipmap.textures.size) {
                        topResults.add(
                            dispatchEdgeDetect(
                                mipmap.textures[idx],
                                Edge.TOP,
                                threshold
                            )
                        )
                    }
                }

                // Bottom edge: last row of tiles
                for (col in 0 until mipmap.tilesCols) {
                    val idx = (mipmap.tilesRows - 1) * mipmap.tilesCols + col
                    if (idx < mipmap.textures.size) {
                        bottomResults.add(
                            dispatchEdgeDetect(
                                mipmap.textures[idx],
                                Edge.BOTTOM,
                                threshold
                            )
                        )
                    }
                }

                val job = launch {
                    while (true) {
                        instance.processEvents()
                        delay(1.milliseconds)
                    }
                }

                try {
                    // Aggregate results per edge direction
                    val edges = listOf(
                        leftResults.awaitAll(),
                        rightResults.awaitAll(),
                        topResults.awaitAll(),
                        bottomResults.awaitAll()
                    )

                    // Collect solid edges with their colors
                    val solidEdges = mutableListOf<Int>()

                    for (edgeTiles in edges) {
                        if (edgeTiles.isEmpty()) continue

                        // Check if all tiles in this edge are solid
                        val allSolid = edgeTiles.all { it.isSolid }
                        if (!allSolid) continue

                        // Calculate weighted average color across all tiles
                        var totalPixels = 0
                        var sumR = 0f
                        var sumG = 0f
                        var sumB = 0f

                        for (result in edgeTiles) {
                            totalPixels += result.total
                            sumR += result.sumR
                            sumG += result.sumG
                            sumB += result.sumB
                        }

                        if (totalPixels > 0) {
                            val avgR = ((sumR / totalPixels) * 255).toInt().coerceIn(0, 255)
                            val avgG = ((sumG / totalPixels) * 255).toInt().coerceIn(0, 255)
                            val avgB = ((sumB / totalPixels) * 255).toInt().coerceIn(0, 255)
                            solidEdges.add((avgR shl 16) or (avgG shl 8) or avgB)
                        }
                    }

                    // Rule: any white edge -> white
                    for (color in solidEdges) {
                        val r = (color shr 16) and 0xFF
                        val g = (color shr 8) and 0xFF
                        val b = color and 0xFF
                        // Check if close to white (threshold ~13 out of 255, i.e. 0.05 * 255)
                        if (r >= 242 && g >= 242 && b >= 242) {
                            return@coroutineScope 0xFFFFFFFF.toInt()
                        }
                    }

                    // Rule: any color edge -> that color (add alpha)
                    if (solidEdges.isNotEmpty()) {
                        return@coroutineScope 0xFF000000.toInt() or solidEdges.first()
                    }

                    // Rule: else -> white
                    0xFFFFFFFF.toInt()
                } finally {
                    job.cancel()
                }
            }
        }

        private enum class Edge { LEFT, RIGHT, TOP, BOTTOM }

        private data class EdgeResult(
            val isSolid: Boolean,
            val total: Int,
            val sumR: Float,
            val sumG: Float,
            val sumB: Float
        )

        private val edgeDetectModule by lazy {
            device.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(EDGE_DETECT_SHADER)
                )
            )
        }

        private val pipelineEdgeLeft by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = edgeDetectModule, entryPoint = "edge_left")
                )
            )
        }

        private val pipelineEdgeRight by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = edgeDetectModule, entryPoint = "edge_right")
                )
            )
        }

        private val pipelineEdgeTop by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = edgeDetectModule, entryPoint = "edge_top")
                )
            )
        }

        private val pipelineEdgeBottom by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = edgeDetectModule, entryPoint = "edge_bottom")
                )
            )
        }

        private val edgeUniformByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
        }
        private val edgeInitByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
        }

        private fun dispatchEdgeDetect(
            texture: GPUTexture,
            edge: Edge,
            threshold: Float
        ): Deferred<EdgeResult> {
            val pipeline = when (edge) {
                Edge.LEFT -> pipelineEdgeLeft
                Edge.RIGHT -> pipelineEdgeRight
                Edge.TOP -> pipelineEdgeTop
                Edge.BOTTOM -> pipelineEdgeBottom
            }

            val uniformBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                )
            )
            // Result: sum_r, sum_g, sum_b, total, sum_sq_r, sum_sq_g, sum_sq_b, padding (8 x u32 = 32 bytes)
            val resultBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 32,
                    usage = BufferUsage.Storage or BufferUsage.CopySrc or BufferUsage.CopyDst
                )
            )
            val stagingBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 32, usage = BufferUsage.CopyDst or BufferUsage.MapRead
                )
            )

            val byteBuffer = edgeUniformByteBuffer.get()
            byteBuffer.clear()
            byteBuffer.putFloat(threshold)
            byteBuffer.putInt(texture.width)
            byteBuffer.putInt(texture.height)
            byteBuffer.putInt(0)
            byteBuffer.flip()
            device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            // Initialize result buffer to zeros
            val initBuffer = edgeInitByteBuffer.get()
            initBuffer.clear()
            repeat(8) { initBuffer.putInt(0) }
            initBuffer.flip()
            device.queue.writeBuffer(resultBuffer, 0L, initBuffer)

            val encoder = device.createCommandEncoder()
            val targetView = texture.createView()
            val pass = encoder.beginComputePass()
            pass.setPipeline(pipeline)
            pass.setTransientBindGroup(
                0, device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, textureView = targetView),
                            GPUBindGroupEntry(1, buffer = resultBuffer),
                            GPUBindGroupEntry(2, buffer = uniformBuffer),
                        )
                    )
                )
            )

            val dispatchSize = when (edge) {
                Edge.LEFT, Edge.RIGHT -> ceil(texture.height / 64.0).toInt()
                Edge.TOP, Edge.BOTTOM -> ceil(texture.width / 64.0).toInt()
            }
            pass.dispatchWorkgroups(dispatchSize)
            pass.endAndRelease(targetView)

            encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 32)
            device.queue.submitAndRelease(encoder)

            val res = CompletableDeferred<EdgeResult>()

            stagingBuffer.mapAsync(
                MapMode.Read,
                0,
                32,
                { it.run() },
                object : GPURequestCallback<Unit> {
                    override fun onResult(result: Unit) {
                        try {
                            val output = stagingBuffer.getConstMappedRange()
                            output.order(ByteOrder.nativeOrder())

                            // Fixed point 16.16 -> float
                            val sumR = output.getInt(0) / 65536f
                            val sumG = output.getInt(4) / 65536f
                            val sumB = output.getInt(8) / 65536f
                            val total = output.getInt(12)
                            val sumSqR = output.getInt(16) / 65536f
                            val sumSqG = output.getInt(20) / 65536f
                            val sumSqB = output.getInt(24) / 65536f

                            stagingBuffer.unmap()

                            if (total == 0) {
                                res.complete(EdgeResult(false, 0, 0f, 0f, 0f))
                            } else {
                                // Calculate average color
                                val avgR = sumR / total
                                val avgG = sumG / total
                                val avgB = sumB / total

                                // Calculate variance: E[X^2] - E[X]^2
                                val varR = (sumSqR / total) - (avgR * avgR)
                                val varG = (sumSqG / total) - (avgG * avgG)
                                val varB = (sumSqB / total) - (avgB * avgB)

                                // Edge is solid if variance is below threshold^2
                                val maxVar = maxOf(varR, varG, varB)
                                val isSolid = maxVar < threshold * threshold

                                res.complete(EdgeResult(isSolid, total, sumR, sumG, sumB))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error reading edge detect result", e)
                            res.complete(EdgeResult(false, 1, 0f, 0f, 0f))
                        } finally {
                            uniformBuffer.destroyAndRelease()
                            resultBuffer.destroyAndRelease()
                            stagingBuffer.destroyAndRelease()
                        }
                    }

                    override fun onError(exception: Exception) {
                        Log.e(TAG, "Error in edge detect mapAsync", exception)
                        res.complete(EdgeResult(false, 1, 0f, 0f, 0f))
                        uniformBuffer.destroyAndRelease()
                        resultBuffer.destroyAndRelease()
                        stagingBuffer.destroyAndRelease()
                    }
                })

            return res
        }

        private const val EDGE_DETECT_SHADER = """
struct Params {
    threshold: f32,
    width: u32,
    height: u32,
    padding: u32,
}

struct Result {
    sum_r: atomic<u32>,
    sum_g: atomic<u32>,
    sum_b: atomic<u32>,
    total: atomic<u32>,
    sum_sq_r: atomic<u32>,
    sum_sq_g: atomic<u32>,
    sum_sq_b: atomic<u32>,
    padding: u32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: Result;
@group(0) @binding(2) var<uniform> params: Params;

fn to_fixed(f: f32) -> u32 {
    return u32(clamp(f, 0.0, 1.0) * 65536.0);
}

fn process_pixel(color: vec3<f32>) {
    // Accumulate color sum and sum of squares for variance calculation
    atomicAdd(&result.sum_r, to_fixed(color.r));
    atomicAdd(&result.sum_g, to_fixed(color.g));
    atomicAdd(&result.sum_b, to_fixed(color.b));
    atomicAdd(&result.total, 1u);
    atomicAdd(&result.sum_sq_r, to_fixed(color.r * color.r));
    atomicAdd(&result.sum_sq_g, to_fixed(color.g * color.g));
    atomicAdd(&result.sum_sq_b, to_fixed(color.b * color.b));
}

@compute @workgroup_size(64)
fn edge_left(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.height) { return; }
    let color = textureLoad(input_tex, vec2<i32>(0, i32(id.x)), 0).rgb;
    process_pixel(color);
}

@compute @workgroup_size(64)
fn edge_right(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.height) { return; }
    let color = textureLoad(input_tex, vec2<i32>(i32(params.width) - 1, i32(id.x)), 0).rgb;
    process_pixel(color);
}

@compute @workgroup_size(64)
fn edge_top(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width) { return; }
    let color = textureLoad(input_tex, vec2<i32>(i32(id.x), 0), 0).rgb;
    process_pixel(color);
}

@compute @workgroup_size(64)
fn edge_bottom(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width) { return; }
    let color = textureLoad(input_tex, vec2<i32>(i32(id.x), i32(params.height) - 1), 0).rgb;
    process_pixel(color);
}
"""

        /** Whole-texture bounding box in one dispatch, reduced per workgroup then globally. */
        private const val TRIM_SHADER = """
struct Params {
    background: vec3<f32>,
    threshold: f32,
}

struct TrimResult {
    min_x: atomic<u32>,
    min_y: atomic<u32>,
    max_x: atomic<u32>,
    max_y: atomic<u32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: TrimResult;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_min_x: atomic<u32>;
var<workgroup> wg_min_y: atomic<u32>;
var<workgroup> wg_max_x: atomic<u32>;
var<workgroup> wg_max_y: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    if (local_invocation_index == 0u) {
        atomicStore(&wg_min_x, 0xFFFFFFFFu);
        atomicStore(&wg_min_y, 0xFFFFFFFFu);
        atomicStore(&wg_max_x, 0u);
        atomicStore(&wg_max_y, 0u);
    }
    workgroupBarrier();

    let dims = vec2<i32>(textureDimensions(input_tex));
    let coords = vec2<i32>(global_id.xy);
    let in_bounds = coords.x < dims.x && coords.y < dims.y;

    if (in_bounds) {
        let color = textureLoad(input_tex, coords, 0);
        let pixel_rgb = color.rgb * color.a + params.background.rgb * (1.0 - color.a);

        let diff = abs(pixel_rgb - params.background.rgb);
        let is_foreground = (diff.r > params.threshold) ||
                            (diff.g > params.threshold) ||
                            (diff.b > params.threshold);

        if (is_foreground) {
            atomicMin(&wg_min_x, u32(coords.x));
            atomicMin(&wg_min_y, u32(coords.y));
            atomicMax(&wg_max_x, u32(coords.x));
            atomicMax(&wg_max_y, u32(coords.y));
        }
    }

    workgroupBarrier();

    if (local_invocation_index == 0u) {
        let w_min_x = atomicLoad(&wg_min_x);
        let w_min_y = atomicLoad(&wg_min_y);
        let w_max_x = atomicLoad(&wg_max_x);
        let w_max_y = atomicLoad(&wg_max_y);

        if (w_min_x != 0xFFFFFFFFu) {
            atomicMin(&result.min_x, w_min_x);
            atomicMin(&result.min_y, w_min_y);
            atomicMax(&result.max_x, w_max_x);
            atomicMax(&result.max_y, w_max_y);
        }
    }
}
"""

        /**
         * One edge per entry point, a thread per row or column, stopping at its first foreground
         * pixel. Used for multi-tile images, where each tile only contributes the edges it owns.
         */
        private const val TRIM_SHADER_SINGLE = """
struct Params {
    background: vec3<f32>,
    threshold: f32,
}

struct TrimResult {
    min_x: atomic<u32>,
    min_y: atomic<u32>,
    max_x: atomic<u32>,
    max_y: atomic<u32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: TrimResult;
@group(0) @binding(2) var<uniform> params: Params;

fn is_foreground(coords: vec2<i32>, dims: vec2<i32>) -> bool {
    let color = textureLoad(input_tex, coords, 0);
    let pixel_rgb = color.rgb * color.a + params.background.rgb * (1.0 - color.a);


    let diff = abs(pixel_rgb - params.background.rgb);
    return (diff.r > params.threshold) ||
           (diff.g > params.threshold) ||
           (diff.b > params.threshold);
}

@compute @workgroup_size(64, 1, 1)
fn find_left(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let x = i32(global_id.x); // Thread index represents a column
    if (x >= dims.x) { return; }

    for (var y = 0; y < dims.y; y = y + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMin(&result.min_x, u32(x));
            break; // Early exit: first foreground pixel found in this column
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_right(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let x = i32(global_id.x); // Thread index represents a column
    if (x >= dims.x) { return; }

    for (var y = 0; y < dims.y; y = y + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMax(&result.max_x, u32(x));
            break; // Early exit
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_top(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let y = i32(global_id.x); // Thread index represents a row index
    if (y >= dims.y) { return; }

    for (var x = 0; x < dims.x; x = x + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMin(&result.min_y, u32(y));
            break; // Early exit
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_bottom(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let y = i32(global_id.x); // Thread index represents a row index
    if (y >= dims.y) { return; }

    for (var x = 0; x < dims.x; x = x + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMax(&result.max_y, u32(y));
            break; // Early exit
        }
    }
}
"""

        private val pipelineAll: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(
                        device.createShaderModule(
                            GPUShaderModuleDescriptor(
                                shaderSourceWGSL = GPUShaderSourceWGSL(TRIM_SHADER)
                            )
                        )
                    )
                )
            )
        }

        private val singleModule by lazy {
            device.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(TRIM_SHADER_SINGLE)
                )
            )
        }

        private val pipelineLeft: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_left")
                )
            )
        }

        private val pipelineRight: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_right")
                )
            )
        }

        private val pipelineTop: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_top")
                )
            )
        }

        private val pipelineBottom: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_bottom")
                )
            )
        }

        /**
         * Find trim bounds for multiple background colors.
         * Runs trim for each color and returns the smallest rect (tightest trim).
         * @param colors List of RGB colors as FloatArrays [r, g, b] with values in 0-1 range
         */
        suspend fun find(image: Image, colors: List<FloatArray>, threshold: Float): Rect {
            require(colors.isNotEmpty()) { "colors must not be empty" }
            require(colors.all { it.size >= 3 }) { "each color must have at least 3 elements [r, g, b]" }

            return WebGpuRenderer.withContext {
                findInContext(image, colors, threshold)
            }
        }

        /**
         * Find trim bounds for multiple background colors when already inside a GPU context.
         * Runs trim for each color and returns the smallest rect (tightest trim).
         */
        suspend fun findInContext(
            image: Image,
            colors: List<FloatArray>,
            threshold: Float
        ): Rect {
            val results = colors.map { color ->
                findInContext(image, color[0], color[1], color[2], threshold)
            }

            // Return the smallest rect (tightest trim)
            return results.minByOrNull { it.width().toLong() * it.height() }
                ?: Rect(0, 0, image.width, image.height)
        }

        /**
         * Find trim bounds for a single color.
         */
        suspend fun find(image: Image, r: Float, g: Float, b: Float, threshold: Float): Rect {
            return WebGpuRenderer.withContext {
                findInContext(image, r, g, b, threshold)
            }
        }

        /**
         * Find trim bounds when already inside a GPU context.
         * Returns full bounds (0,0,width,height) if trim detection fails.
         */
        suspend fun findInContext(
            image: Image,
            r: Float,
            g: Float,
            b: Float,
            threshold: Float
        ): Rect {
            if (image.mipmaps.isEmpty()) {
                Log.w(TAG, "findInContext: image has no mipmaps, returning full bounds")
                return Rect(0, 0, image.width, image.height)
            }

            val mipmap = image.mipmaps[0]

            if (mipmap.textures.isEmpty()) {
                Log.w(TAG, "findInContext: mipmap has no textures, returning full bounds")
                return Rect(0, 0, image.width, image.height)
            }

            return try {
                if (mipmap.tilesCols == 1 && mipmap.tilesRows == 1) {
                    findSingleTile(mipmap.textures[0], r, g, b, threshold)
                } else {
                    findMultiTile(mipmap, r, g, b, threshold)
                }
            } catch (e: Exception) {
                Log.e(TAG, "findInContext: error during trim detection, returning full bounds", e)
                Rect(0, 0, image.width, image.height)
            }
        }

        private suspend fun findSingleTile(
            texture: GPUTexture,
            r: Float, g: Float, b: Float, threshold: Float
        ): Rect = coroutineScope {
            val res = dispatchTrimCompute(texture, pipelineAll, r, g, b, threshold)

            val job = launch {
                while (true) {
                    instance.processEvents()
                    delay(1.milliseconds)
                }
            }

            try {
                res.await()
            } finally {
                job.cancel()
            }
        }

        private suspend fun findMultiTile(
            mipmap: Mipmap,
            r: Float, g: Float, b: Float, threshold: Float
        ): Rect = coroutineScope {
            val left = mutableListOf<Deferred<Rect>>()
            val right = mutableListOf<Deferred<Rect>>()
            val top = mutableListOf<Deferred<Rect>>()
            val bottom = mutableListOf<Deferred<Rect>>()

            for (row in 0 until mipmap.tilesRows) {
                val leftIdx = row * mipmap.tilesCols
                val rightIdx = row * mipmap.tilesCols + mipmap.tilesCols - 1

                if (leftIdx < mipmap.textures.size) {
                    left.add(
                        dispatchTrimCompute(
                            mipmap.textures[leftIdx], pipelineLeft, r, g, b, threshold
                        )
                    )
                }
                if (rightIdx < mipmap.textures.size) {
                    right.add(
                        dispatchTrimCompute(
                            mipmap.textures[rightIdx], pipelineRight, r, g, b, threshold
                        )
                    )
                }
            }

            for (col in 0 until mipmap.tilesCols) {
                val topIdx = col
                val bottomIdx = (mipmap.tilesRows - 1) * mipmap.tilesCols + col

                if (topIdx < mipmap.textures.size) {
                    top.add(
                        dispatchTrimCompute(
                            mipmap.textures[topIdx], pipelineTop, r, g, b, threshold
                        )
                    )
                }
                if (bottomIdx < mipmap.textures.size) {
                    bottom.add(
                        dispatchTrimCompute(
                            mipmap.textures[bottomIdx], pipelineBottom, r, g, b, threshold
                        )
                    )
                }
            }

            val job = launch {
                while (true) {
                    instance.processEvents()
                    delay(1.milliseconds)
                }
            }

            try {
                val leftResults = left.awaitAll()
                val topResults = top.awaitAll()
                val rightResults = right.awaitAll()
                val bottomResults = bottom.awaitAll()

                Rect(
                    leftResults.minOfOrNull { it.left } ?: 0,
                    topResults.minOfOrNull { it.top } ?: 0,
                    (rightResults.maxOfOrNull { it.right } ?: mipmap.tilesize) +
                            mipmap.tilesize * (mipmap.tilesCols - 1),
                    (bottomResults.maxOfOrNull { it.bottom } ?: mipmap.tilesize) +
                            mipmap.tilesize * (mipmap.tilesRows - 1),
                )
            } finally {
                job.cancel()
            }
        }

        private val trimUniformByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
        }
        private val trimInitByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
        }

        /**
         * Dispatch a trim compute operation. Creates fresh GPU buffers for each call
         * to avoid concurrency issues with buffer reuse during async operations.
         */
        private fun dispatchTrimCompute(
            texture: GPUTexture,
            pipeline: GPUComputePipeline,
            r: Float, g: Float, b: Float, threshold: Float
        ): Deferred<Rect> {
            val uniformBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                )
            )
            val resultBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16,
                    usage = BufferUsage.Storage or BufferUsage.CopySrc or BufferUsage.CopyDst
                )
            )
            val stagingBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16, usage = BufferUsage.CopyDst or BufferUsage.MapRead
                )
            )

            val byteBuffer = trimUniformByteBuffer.get()
            byteBuffer.clear()
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
            byteBuffer.putFloat(threshold)
            byteBuffer.flip()
            device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            val initBuffer = trimInitByteBuffer.get()
            initBuffer.clear()
            initBuffer.putInt(texture.width)
            initBuffer.putInt(texture.height)
            initBuffer.putInt(0)
            initBuffer.putInt(0)
            initBuffer.flip()
            device.queue.writeBuffer(resultBuffer, 0L, initBuffer)

            val encoder = device.createCommandEncoder()
            val targetView = texture.createView()
            val pass = encoder.beginComputePass()
            pass.setPipeline(pipeline)
            pass.setTransientBindGroup(
                0, device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, textureView = targetView),
                            GPUBindGroupEntry(1, buffer = resultBuffer),
                            GPUBindGroupEntry(2, buffer = uniformBuffer),
                        )
                    )
                )
            )

            pass.dispatchWorkgroups(
                ceil(texture.width / 8.0).toInt(), ceil(texture.height / 8.0).toInt()
            )
            pass.endAndRelease(targetView)

            encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 16)

            device.queue.submitAndRelease(encoder)

            val res = CompletableDeferred<Rect>()

            stagingBuffer.mapAsync(
                MapMode.Read,
                0,
                16,
                { it.run() },
                object : GPURequestCallback<Unit> {
                    override fun onResult(result: Unit) {
                        try {
                            val output = stagingBuffer.getConstMappedRange()
                            output.order(ByteOrder.nativeOrder())
                            val rect = Rect(
                                output.getInt(0),
                                output.getInt(4),
                                (output.getInt(8) + 1).coerceAtMost(texture.width),
                                (output.getInt(12) + 1).coerceAtMost(texture.height),
                            )
                            stagingBuffer.unmap()
                            res.complete(rect)
                        } catch (e: Exception) {
                            Log.e(TAG, "Error reading trim result", e)
                            res.complete(Rect(0, 0, texture.width, texture.height))
                        } finally {
                            uniformBuffer.destroyAndRelease()
                            resultBuffer.destroyAndRelease()
                            stagingBuffer.destroyAndRelease()
                        }
                    }

                    override fun onError(exception: Exception) {
                        Log.e(TAG, "Error in trim mapAsync", exception)
                        res.complete(Rect(0, 0, texture.width, texture.height))
                        uniformBuffer.destroyAndRelease()
                        resultBuffer.destroyAndRelease()
                        stagingBuffer.destroyAndRelease()
                    }
                })

            return res
        }
    }
}
