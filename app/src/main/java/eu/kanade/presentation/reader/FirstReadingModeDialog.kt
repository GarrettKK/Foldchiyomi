package eu.kanade.presentation.reader

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.vectorResource
import androidx.compose.ui.tooling.preview.PreviewLightDark
import androidx.compose.ui.unit.dp
import eu.kanade.presentation.components.AdaptiveSheet
import eu.kanade.presentation.theme.TachiyomiPreviewTheme
import eu.kanade.tachiyomi.ui.reader.setting.ReadingMode
import mihon.icons.materialsymbols.MaterialSymbols
import mihon.icons.materialsymbols.rounded.Check
import tachiyomi.i18n.MR
import tachiyomi.presentation.core.components.SettingsIconGrid
import tachiyomi.presentation.core.components.SettingsItemsPaddings
import tachiyomi.presentation.core.components.material.IconToggleButton
import tachiyomi.presentation.core.components.material.padding
import tachiyomi.presentation.core.i18n.stringResource

/**
 * Asked the first time a series is opened without a reading mode of its own, so manga and
 * left-to-right comics can share a library without a trip to the settings for each one.
 */
@Composable
fun FirstReadingModeDialog(
    onDismissRequest: () -> Unit,
    initial: ReadingMode,
    onApply: (ReadingMode) -> Unit,
    onStopAsking: () -> Unit,
) {
    AdaptiveSheet(onDismissRequest = onDismissRequest) {
        DialogContent(initial = initial, onApply = onApply, onStopAsking = onStopAsking)
    }
}

@Composable
private fun DialogContent(
    initial: ReadingMode,
    onApply: (ReadingMode) -> Unit,
    onStopAsking: () -> Unit,
) {
    var selected by remember { mutableStateOf(initial) }

    Column(modifier = Modifier.padding(vertical = 16.dp)) {
        Column(
            modifier = Modifier.padding(horizontal = SettingsItemsPaddings.Horizontal),
            verticalArrangement = Arrangement.spacedBy(MaterialTheme.padding.small),
        ) {
            Text(
                text = stringResource(MR.strings.first_reading_mode_title),
                style = MaterialTheme.typography.titleLarge,
            )
            Text(
                text = stringResource(MR.strings.first_reading_mode_summary),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SettingsIconGrid(MR.strings.pref_category_reading_mode) {
            items(ReadingMode.entries - ReadingMode.DEFAULT) { mode ->
                IconToggleButton(
                    checked = mode == selected,
                    onCheckedChange = { selected = mode },
                    modifier = Modifier.fillMaxWidth(),
                    imageVector = ImageVector.vectorResource(mode.iconRes),
                    title = stringResource(mode.stringRes),
                )
            }
        }

        Row(
            modifier = Modifier.padding(horizontal = SettingsItemsPaddings.Horizontal),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onStopAsking) {
                Text(text = stringResource(MR.strings.action_dont_ask_again))
            }

            Spacer(modifier = Modifier.weight(1f))

            FilledTonalButton(onClick = { onApply(selected) }) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(MaterialTheme.padding.small),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(imageVector = MaterialSymbols.Rounded.Check, contentDescription = null)
                    Text(text = stringResource(MR.strings.action_apply))
                }
            }
        }
    }
}

@PreviewLightDark
@Composable
private fun DialogContentPreview() {
    TachiyomiPreviewTheme {
        Surface {
            DialogContent(initial = ReadingMode.RIGHT_TO_LEFT, onApply = {}, onStopAsking = {})
        }
    }
}
