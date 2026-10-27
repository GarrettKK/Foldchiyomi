package eu.kanade.tachiyomi.ui.reader.viewer.webgpu

import android.content.Context
import android.media.AudioAttributes
import android.media.SoundPool
import eu.kanade.tachiyomi.R

/**
 * The soft paper flip a page turn can make - see [eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences.pageTurnSound].
 *
 * Plays as a system sound, like a keyboard's clicks: it follows the system sound volume, and
 * silent mode mutes it, so reading in bed next to someone asleep stays quiet.
 */
class PageTurnSound(context: Context) {
    private val pool = SoundPool.Builder()
        .setMaxStreams(2)
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build(),
        )
        .build()

    @Volatile
    private var loaded = false

    private val id: Int

    init {
        pool.setOnLoadCompleteListener { _, _, status -> loaded = status == 0 }
        id = pool.load(context, R.raw.page_flip, 1)
    }

    /** From any thread; nothing plays until the clip has loaded. */
    fun play() {
        if (loaded) pool.play(id, VOLUME, VOLUME, 1, 0, 1f)
    }

    fun release() {
        loaded = false
        pool.release()
    }

    private companion object {
        const val VOLUME = 0.7f
    }
}
