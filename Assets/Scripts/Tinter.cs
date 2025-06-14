using UnityEngine;
using UnityEngine.UI;

public class Tinter {
    static public void Hierarchy(Transform root, Color color) {
        foreach (Transform child in root) {
            // UI Tinting
            Graphic ui = child.GetComponent<Graphic>();
            if ( ui != null ) {
                ui.color *= color;
            }

            // 3D Renderer Tinting
            Renderer renderer = child.GetComponent<Renderer>();
            if ( renderer != null ) {
                // Clone the material to avoid affecting other instances
                Material mat = renderer.material;

                // Handle shaders that use _Color or _BaseColor
                if ( mat.HasProperty("_Color") ) {
                    mat.color *= color;
                } else if ( mat.HasProperty("_BaseColor") ) {
                    mat.SetColor("_BaseColor", mat.GetColor("_BaseColor") * color);
                }
            }

            // Recurse
            Hierarchy(child, color);
        }
    }
}