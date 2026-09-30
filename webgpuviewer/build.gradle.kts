import org.gradle.api.file.ArchiveOperations
import org.gradle.api.file.FileSystemOperations
import javax.inject.Inject

/*
 * The WebGPU page viewer, carried in-tree from https://github.com/mpreg-ca/webgpuviewer (tag 49,
 * MIT - see LICENSE) so the reader can have its own page curl, TransitionCurl.
 *
 * Kotlin only: its small native library, libresize.so, is taken prebuilt from the same release's
 * published AAR rather than built here, so no NDK or CMake is needed.
 */
plugins {
    alias(mihonx.plugins.android.library)
    alias(mihonx.plugins.compose)
}

android {
    namespace = "ca.mpreg.webgpuviewer"

    defaultConfig {
        consumerProguardFiles("consumer-rules.pro")
    }
}

val upstreamAar: Configuration by configurations.creating {
    isTransitive = false
    isCanBeConsumed = false
}

dependencies {
    upstreamAar(variantOf(libs.webgpuviewer) { artifactType("aar") })

    api(libs.androidx.webgpu)
    implementation(libs.androidx.annotation)
    implementation(libs.androidx.core)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.kotlinx.coroutines.core)
}

/** Unpacks the release's prebuilt libresize.so, one per ABI, as a jniLibs directory. */
abstract class ExtractNativeLibsTask : DefaultTask() {
    @get:InputFiles
    abstract val aar: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val archives: ArchiveOperations

    @get:Inject
    abstract val fileSystem: FileSystemOperations

    @TaskAction
    fun extract() {
        val trees = aar.files.map { archives.zipTree(it) }
        fileSystem.sync {
            from(trees) {
                include("jni/*/libresize.so")
                eachFile { path = path.removePrefix("jni/") }
            }
            includeEmptyDirs = false
            into(outputDir)
        }
    }
}

val extractNativeLibs = tasks.register<ExtractNativeLibsTask>("extractUpstreamNativeLibs") {
    aar.from(upstreamAar)
    outputDir.set(layout.buildDirectory.dir("generated/upstreamJniLibs"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.jniLibs?.addGeneratedSourceDirectory(extractNativeLibs) { it.outputDir }
    }
}
