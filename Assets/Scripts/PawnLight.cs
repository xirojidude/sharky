
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class PawnLight : UdonSharpBehaviour
{
    private Transform Piece;
    private bool dropped = false;

    void Start()
    {
       // Piece = GetComponent<Transform>();
    }

    private void Update()
    {
        if (dropped) {
            float row = Mathf.Floor(this.transform.localPosition.z + .5f); 
            float col = Mathf.Floor(this.transform.localPosition.x + .5f);

            this.transform.localPosition = new Vector3(col,0f,row);
            this.transform.localRotation = new Quaternion(0f,0f,0f,0f);
            dropped = false;
        }
    }

    private void onDrop()
    {
        Debug.Log("Piece dropped");
        this.transform.localRotation = new Quaternion(0f,0f,0f,0f);
        dropped = true;
    }
}
