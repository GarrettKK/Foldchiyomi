plugins {
    alias(mihonx.plugins.android.library)
    alias(mihonx.plugins.compose)
}

android {
    namespace = "ca.mpreg.webgpuviewer"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("proguard-rules.txt")

        externalNativeBuild {
            cmake {
                cppFlags("-O3 -flto")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    api(libs.androidx.annotation)
    api(libs.androidx.core)
    api(libs.androidx.compose.foundation)
    api(libs.androidx.webgpu)

    implementation(libs.androidx.annotation.experimental)
    implementation(libs.kotlinx.coroutines.core)
}
