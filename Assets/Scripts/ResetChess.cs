
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class ResetChess : UdonSharpBehaviour
{
    public Transform[] WhitePieces = new Transform[16];
    public Transform[] BlackPieces = new Transform[16];
    public Transform Board;

    void Start()
    {
        
    }

    public void Interact()
    {
        // Reset Pieces
        for (int i=0; i<16; i++)
        {
            float col = i % 8f;
            float row = Mathf.Floor( i*1f / 8f);
            WhitePieces[i].localPosition = new Vector3(col,0f,7f-row); 
            
            BlackPieces[i].localPosition = new Vector3(col,0f,row); 

        }
    }
}
