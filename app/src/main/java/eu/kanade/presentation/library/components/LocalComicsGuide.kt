package eu.kanade.presentation.library.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.PreviewLightDark
import androidx.compose.ui.unit.dp
import eu.kanade.presentation.theme.TachiyomiPreviewTheme
import mihon.icons.materialsymbols.MaterialSymbols
import mihon.icons.materialsymbols.rounded.Folder
import tachiyomi.i18n.MR
import tachiyomi.presentation.core.i18n.stringResource

/**
 * The empty library: how to read comics already on the phone, shown right where they will
 * appear rather than behind a link - for readers who come for their own files, not for sources.
 */
@Composable
fun LocalComicsGuide(
    onOpenStorageSettings: () -> Unit,
    onOpenLocalSource: () -> Unit,
    onOpenGettingStarted: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Column(
            modifier = Modifier.widthIn(max = 480.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = stringResource(MR.strings.information_empty_library),
                style = MaterialTheme.typography.headlineSmall,
            )
            Text(
                text = stringResource(MR.strings.local_guide_intro),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Step(1, stringResource(MR.strings.local_guide_step_storage))
            Step(2, stringResource(MR.strings.local_guide_step_series))
            Step(3, stringResource(MR.strings.local_guide_step_files))

            FolderTree()

            Text(
                text = stringResource(MR.strings.local_guide_formats),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onOpenLocalSource) {
                    Text(stringResource(MR.strings.local_guide_open_local))
                }
                OutlinedButton(onClick = onOpenStorageSettings) {
                    Text(stringResource(MR.strings.local_guide_storage_settings))
                }
            }

            Text(
                text = stringResource(MR.strings.local_guide_sources),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = onOpenGettingStarted) {
                Text(stringResource(MR.strings.getting_started_guide))
            }
        }
    }
}

@Composable
private fun Step(number: Int, text: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Surface(
            shape = RoundedCornerShape(50),
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.size(28.dp),
        ) {
            Text(
                text = number.toString(),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                modifier = Modifier.padding(top = 4.dp),
                textAlign = TextAlign.Center,
            )
        }
        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.weight(1f),
        )
    }
}

/** What the three steps leave on disk, drawn as the folder tree a file manager would show. */
@Composable
private fun FolderTree() {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            TreeRow(0, stringResource(MR.strings.local_guide_tree_storage), folder = true)
            TreeRow(1, "local", folder = true)
            TreeRow(2, "Batman Year One", folder = true)
            TreeRow(3, "Chapter 1.cbz", folder = false)
            TreeRow(3, "Chapter 2.cbz", folder = false)
            TreeRow(2, "Dungeon Meshi", folder = true)
            TreeRow(3, "Volume 01.cbr", folder = false)
        }
    }
}

@Composable
private fun TreeRow(depth: Int, name: String, folder: Boolean) {
    Row(
        modifier = Modifier.padding(start = (depth * 20).dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (folder) {
            Icon(
                imageVector = MaterialSymbols.Rounded.Folder,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(18.dp),
            )
        } else {
            Text(
                text = "•",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 5.dp),
            )
        }
        Text(
            text = name,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = if (folder) null else FontFamily.Monospace,
        )
    }
}

@PreviewLightDark
@Composable
private fun LocalComicsGuidePreview() {
    TachiyomiPreviewTheme {
        Surface {
            LocalComicsGuide(onOpenStorageSettings = {}, onOpenLocalSource = {}, onOpenGettingStarted = {})
        }
    }
}
